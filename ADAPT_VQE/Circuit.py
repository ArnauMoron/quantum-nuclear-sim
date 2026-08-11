import numpy as np
from collections import Counter, defaultdict

from openfermion.ops import FermionOperator, QubitOperator
from openfermion.transforms import jordan_wigner
import os

from qibo import gates
from qibo.models.circuit import Circuit
from qibo.result import MeasurementOutcomes

from CORE.Nucleus import Nucleus
from CORE.CircuitBuilder import CircuitBuilder

from scipy.optimize import minimize

import io
import contextlib


class Circuits_Composer(CircuitBuilder):
    """
    Translates the ansatz's excitation operators into executable Qibo circuits and
    measures the energy/gradients, for either the fermionic or the HCB (hard-core boson)
    encoding.

    This is the ADAPT-VQE-specific circuit-measurement machinery (operator pools,
    ansatz layers, gradient/GGF measurement, HCB energy measurement). The
    infrastructure it shares with the annealing package (reference-state circuit,
    qubit re-basing, the staircase Pauli-exponential builder, the Hermitian/1-body
    Jordan-Wigner conversions) lives in the CircuitBuilder base class instead of
    being duplicated here.
    """

    def __init__(
        self,
        nucleus: str = "Be6",
        shell="p",
        ref_state: int = 0,
        parameters: list = [],
        operators_used: list = [],
        only_ref: bool = False,
        exact: bool = True,
        nshots: int = 1000,
        encoding: str = "fermionic",
        nuc: Nucleus = None,
        statevector_caching: bool = True,
        hop_measurement: str = "givens",
    ):

        # Reuse an already-built Nucleus if the caller has one (e.g. every Ansatz
        # already constructs its own Nucleus before creating this composer), instead
        # of redundantly rebuilding one from the raw data files: re-parsing the data
        # files and rebuilding the whole operator/commutator pool from scratch.
        if nuc is None:
            nuc = Nucleus(nucleus, shell=shell, encoding=encoding)
        else:
            assert (
                nuc.encoding == encoding
            ), "Nucleus passed to Circuits_Composer has a different encoding"

        super().__init__(nuc, ref_state=ref_state, encoding=encoding)

        self.exact = exact
        self.nshots = nshots
        self.f = io.StringIO()

        # When True (the default), exact/noiseless energy, gradient and GGF
        # parameter-offset evaluations simulate the shared ansatz (+offset/shift)
        # prefix once and reuse its cached statevector (via initial_state=) for
        # every small suffix circuit that follows it, instead of re-simulating the
        # whole prefix from scratch for every term/pair. This is a simulator-only
        # speedup: real quantum hardware cannot be handed an arbitrary
        # initial_state= to resume from, it can only execute a circuit's gate
        # sequence starting from |0...0>. Set this to False to bypass every such
        # cached-statevector fast path and always build/execute the full,
        # from-|0...0> circuit for each measurement, i.e. exactly what would need
        # to run on real hardware (or a hardware-faithful simulator/transpiler).
        # See the "Statevector caching vs. hardware-realistic execution" section of
        # the README for details on which methods this affects.
        self.statevector_caching = statevector_caching

        # Selects how the HCB "complex" measurement path (Circuits_Composer.complex
        # == True) measures the non-diagonal XX+YY pair-hopping terms. Both
        # strategies use the same Z-basis reconstruction technique - the term's
        # magnitude is always read from the unrotated Z-basis coherence
        # (2*sqrt(P_01*P_10)) - and only differ in which extra rotation is used to
        # pin down the term's *sign*:
        #   - "givens" (default): rotates each pair with a two-qubit Givens gate
        #     (GIVENS(-pi/4)) before measuring, which diagonalizes XX+YY into
        #     Zp-Zq (paper's "Basis rotation"/BasisRot strategy, Methods, Eq. 17),
        #     and reads the sign off that. Unlike H, this rotation conserves
        #     particle number on the pair.
        #   - "hadamard": the original strategy - rotates each pair into the X
        #     basis (H(A);H(B)) and reads the sign off <XX> (assuming
        #     <XX>=<YY>). Kept for comparison with the paper's "Hadamard"
        #     strategy.
        # Only affects the HCB "complex" path; ignored for the "simple" HCB path and
        # the fermionic encoding, which don't branch on it.
        if hop_measurement not in ("givens", "hadamard"):
            raise ValueError(
                f"hop_measurement must be 'givens' or 'hadamard', got {hop_measurement!r}"
            )
        self.hop_measurement = hop_measurement

        # Jordan-Wigner Pauli decompositions of the antihermitian ansatz excitation
        # operators (Operator_index_to_Pauli, ADAPT-only - unlike the Hermitian/
        # 1-body conversions inherited from CircuitBuilder) and the per-Hamiltonian
        # -term suffix circuits only depend on static configuration
        # (n_qubits/operator_pool), never on operators_used/parameters, so they are
        # safe to cache for the lifetime of this instance.
        self._pauli_cache = {}
        self._term_entries_cache = None

        if encoding == "fermionic":
            ham = nuc.Ham_2_body_contributions()
            operator_pool = nuc.operators

            two_index = []

            for op in ham:
                if len(set(op.ijkl)) == 2:
                    two_index.append(op)

            one_body, monoparticular = nuc.Ham_1_body_contributions()

            self.monoparticular_energies = monoparticular

            self.operator_pool = operator_pool
            self.two_index = two_index
            self.operators_used = operators_used
            self.parameters = parameters
            self.only_ref = only_ref

            self.name = nuc.name
            self.data_folder = os.path.join(f"nuclei/{self.name}_data")

        elif encoding == "HCB":
            self.operators_used = operators_used
            self.parameters = parameters

            self.operator_pool = nuc.ops_hop

            if len(self.ref_state_indexes) != 1 and len(self.ref_state_indexes) != (self.n_qubits - 1):
                self.complex = True
            else:
                self.complex = False

            self.ops_1_body = nuc.ops_1_body
            self.ops_2_body = nuc.ops_2_body
            self.ops_hop = nuc.ops_hop

    def Operator_index_to_Pauli(self, index):
        """Converts i (a†_i a†_j a_k a_l - a†_k a†_l a_i a_j) into a Pauli string over n_qubits.

        The Jordan-Wigner decomposition of a given operator index never changes (it
        only depends on the fixed index tuple and the static n_qubits/qubit_offset),
        but this is called on every single circuit build for every operator already
        in the ansatz, so the result is cached per index for the lifetime of this
        composer instance.
        """
        key = tuple(index)
        cached = self._pauli_cache.get(key)
        if cached is not None:
            return cached

        i, j, k, l = self._local_list(index)

        fermion_op = FermionOperator(f"{i}^ {j}^ {k} {l}", 1j)  # i a†_i a†_j a_k a_l
        fermion_op += FermionOperator(f"{k}^ {l}^ {i} {j}", -1j)  # -i a†_k a†_l a_i a_j

        pauli_op = jordan_wigner(fermion_op)

        filtered_pauli_op = QubitOperator()
        for term, coef in pauli_op.terms.items():
            if all(q < self.n_qubits for q, _ in term):
                filtered_pauli_op += QubitOperator(term, coef)

        self._pauli_cache[key] = filtered_pauli_op
        return filtered_pauli_op

    def Qibo_layer_composer(self):

        if self.encoding == "HCB":
            circ = Circuit(self.n_qubits)

            for i in range(len(self.operators_used)):
                A = self.operators_used[i].A
                B = self.operators_used[i].B
                par = self.parameters[i]
                circ.add(gates.GIVENS(A, B, -par))

            return circ

        # Constructs a quantum circuit layer for given fermionic indices using the staircase algorithm for pauli exponentials.
        circuit = Circuit(self.n_qubits)
        for n in range(len(self.operators_used)):
            pauli_op = self.Operator_index_to_Pauli(self.operators_used[n].ijkl)
            self.Qibo_staircase_pauli_exponential(circuit, pauli_op, self.parameters[n])

        return circuit

    def qibo_3_dif_index(self, j, k):
        j, k = self._local(j), self._local(k)
        circuit = Circuit(self.n_qubits)

        circuit.add(gates.CNOT(k, j))
        circuit.add(gates.H(k))
        circuit.add(gates.CNOT(k, j))

        return circuit

    def qibo_4_dif_index(self, operator_index):
        circuit = Circuit(self.n_qubits)
        i, j, k, l = self._local_list(operator_index)

        circuit.add(gates.CNOT(i, j))
        circuit.add(gates.CNOT(k, i))
        circuit.add(gates.CNOT(l, k))
        circuit.add(gates.H(l))
        circuit.add(gates.CNOT(l, k))
        circuit.add(gates.CNOT(k, i))
        circuit.add(gates.CNOT(i, j))

        return circuit

    def Qibo_all_circuits(self, circ=None):

        if self.encoding == "HCB":
            if circ == None:
                main_circuit = (
                    self.Qibo_ref_state_composer() + self.Qibo_layer_composer()
                )
            else:
                main_circuit = circ

            XX_observables_circ = main_circuit.copy()
            # YY_observables_circ = main_circuit.copy()

            for q in range(self.n_qubits):
                XX_observables_circ.add(gates.H(q))
                # YY_observables_circ.add(gates.RX(q, np.pi / 2))

            return [main_circuit, XX_observables_circ]  # , YY_observables_circ]

        if circ is not None:
            main_circuit = circ
        elif self.only_ref is False:
            main_circuit = self.Qibo_ref_state_composer() + self.Qibo_layer_composer()
        else:
            main_circuit = self.Qibo_ref_state_composer()

        circuits = [[main_circuit, 0, 0]]
        circuits_3 = {}

        for op in self.operator_pool:

            indexes = op.ijkl
            matrix_element = op.H2b

            if len(set(indexes)) == 4:
                circuit = main_circuit + self.qibo_4_dif_index(operator_index=indexes)
                circuits.append([circuit, matrix_element, indexes])

            elif len(set(indexes)) == 3:
                counter = Counter(indexes)
                repetido = [num for num, count in counter.items() if count > 1][0]
                j, k = [num for num in indexes if num != repetido]

                jk_sorted = tuple(sorted([j, k]))

                if jk_sorted in circuits_3:
                    circuits_3[jk_sorted][1].append(matrix_element)
                    circuits_3[jk_sorted][2].append(repetido)
                    circuits_3[jk_sorted][3].append(indexes)
                else:
                    j, k = sorted([j, k])
                    circuit = main_circuit + self.qibo_3_dif_index(j=j, k=k)
                    circuits_3[jk_sorted] = [
                        circuit,
                        [matrix_element],
                        [repetido],
                        [indexes],
                        jk_sorted,
                    ]

        for key in circuits_3.keys():
            circuits.append(circuits_3[key])

        return circuits

    def _get_term_entries(self):
        """
        Builds the list of small per-Hamiltonian-term suffix circuits used by
        _Qibo_energy_from_prefix (fermionic). This decomposition only depends on the
        static self.operator_pool/n_qubits (never on operators_used/parameters, i.e.
        never on which ansatz layer/prefix it will later be measured against), so it
        is computed once and cached for the lifetime of this composer instance
        instead of being rebuilt from scratch on every energy/gradient evaluation.
        """
        if self._term_entries_cache is not None:
            return self._term_entries_cache

        term_entries = []
        circuits_3 = {}

        for op in self.operator_pool:

            indexes = op.ijkl
            matrix_element = op.H2b

            if len(set(indexes)) == 4:
                suffix = self.qibo_4_dif_index(operator_index=indexes)
                term_entries.append([suffix, matrix_element, indexes])

            elif len(set(indexes)) == 3:
                counter = Counter(indexes)
                repetido = [num for num, count in counter.items() if count > 1][0]
                j, k = [num for num in indexes if num != repetido]

                jk_sorted = tuple(sorted([j, k]))

                if jk_sorted in circuits_3:
                    circuits_3[jk_sorted][1].append(matrix_element)
                    circuits_3[jk_sorted][2].append(repetido)
                    circuits_3[jk_sorted][3].append(indexes)
                else:
                    j, k = sorted([j, k])
                    suffix = self.qibo_3_dif_index(j=j, k=k)
                    circuits_3[jk_sorted] = [
                        suffix,
                        [matrix_element],
                        [repetido],
                        [indexes],
                        jk_sorted,
                    ]

        for key in circuits_3.keys():
            term_entries.append(circuits_3[key])

        self._term_entries_cache = term_entries
        return term_entries

    def _Qibo_all_circuits_split(self):
        """
        Same term decomposition as Qibo_all_circuits (fermionic), but returns the
        shared ansatz-prefix circuit separately from the small per-term suffix
        circuits, so the prefix can be simulated once and reused (via initial_state=)
        instead of being re-simulated from scratch for every Hamiltonian term.
        """
        if self.only_ref is False:
            main_circuit = self.Qibo_ref_state_composer() + self.Qibo_layer_composer()
        else:
            main_circuit = self.Qibo_ref_state_composer()

        return main_circuit, self._get_term_entries()

    def _Qibo_energy_from_prefix(self, main_circuit):
        """
        Core of the fermionic, exact energy measurement: simulates main_circuit
        (an arbitrary ansatz/offset/shift prefix) exactly once, then reuses its
        cached statevector (via initial_state=) for every small Hamiltonian-term
        suffix circuit from _get_term_entries(), instead of re-simulating
        main_circuit's gates again for every term. This lets any caller that needs
        the energy for a given prefix (plain ansatz energy, a GGF parameter-offset
        variant, or a gradient parameter-shift variant) reuse the same fast path
        that was previously only wired up for the plain top-level energy() call.
        """
        term_entries = self._get_term_entries()

        result = main_circuit()
        print(result)
        main_state = np.array(result.state())

        E_1 = 0
        E_2_index = 0

        print("Energy,\t\t\tobservable,\t\tamplitude\n")

        for part in self.monoparticular_energies:
            i = int(part)
            eps = self.monoparticular_energies[part]

            E_i = eps * result.probabilities([self._local(i)])[1]

            E_1 += E_i

            if abs(E_i) > 10 ** (-3):
                print(np.round(E_i, 8), "\t\t", i, "\t\t", np.round(eps, 8))

        for op in self.two_index:

            nu_ij = op.H2b
            ij = set(op.ijkl)

            E_i = nu_ij * result.probabilities(set(self._local_list(ij)))[3]
            E_2_index += E_i

            if abs(E_i) > 10 ** (-3):
                print(np.round(E_i, 8), "\t\t", op.ijkl, "\t\t", np.round(nu_ij, 8))

        En_4_index = 0
        En_3_index = 0

        for entry in term_entries:
            suffix = entry[0]
            res = suffix(initial_state=main_state.copy())

            if len(entry) == 3:

                op_amplitude = entry[1]
                op_index = entry[2]

                indices = self.obs_P(op_index)

                bitstrings = []
                for numero in range(2 ** len(indices)):
                    bitstring = bin(numero)[2:]
                    bitstrings.append(bitstring)

                ordenes = indices + self._local_list(op_index)

                E_ijkl = 0

                for bitstring in bitstrings:
                    n = 0
                    for bit in bitstring:
                        n += int(bit)

                    sign = (-1) ** n

                    E_ijkl += (
                        sign
                        * op_amplitude
                        * (
                            res.probabilities(ordenes)[int(bitstring + "1100", 2)]
                            - res.probabilities(ordenes)[int(bitstring + "0011", 2)]
                        )
                    )

                En_4_index += E_ijkl

                if abs(E_ijkl) > 10 ** (-3):
                    print(
                        np.round(E_ijkl, 8),
                        "\t\t",
                        op_index,
                        "\t\t",
                        np.round(op_amplitude, 8),
                    )

            elif len(entry) == 5:

                op_amplitudes = entry[1]
                reps = entry[2]
                j, k = entry[4]
                op_indexs = entry[3]

                for r in range(len(reps)):
                    rep = reps[r]
                    op_index = op_indexs[r]
                    indices = self.obs_P(op_index)

                    op_index = [rep, j, k]
                    op_amplitude = op_amplitudes[r]

                    bitstrings = []
                    for numero in range(2 ** len(indices)):
                        bitstring = bin(numero)[2:]
                        bitstrings.append(bitstring)

                    ordenes = indices + self._local_list(op_index)

                    E_njnk = 0

                    for bitstring in bitstrings:
                        par = 0
                        for bit in bitstring:
                            par += int(bit)

                        sign = (-1) ** par

                        E_njnk += (
                            sign
                            * op_amplitude
                            * (
                                res.probabilities(ordenes)[int(bitstring + "110", 2)]
                                - res.probabilities(ordenes)[int(bitstring + "101", 2)]
                            )
                        )

                    En_3_index += E_njnk

                    if abs(E_njnk) > 10 ** (-3):
                        print(
                            np.round(E_njnk, 8),
                            "\t\t",
                            op_index,
                            "\t\t",
                            np.round(op_amplitude, 8),
                        )

        print(
            "\n-----------------------------------Final Energies-----------------------------------\n"
        )
        print("Monoparticular energies:                 ", E_1)
        print("Two body energies with 2 diferent index: ", E_2_index)
        print("Two body energies with 3 diferent index: ", En_3_index)
        print("Two body energies with 4 diferent index: ", En_4_index)

        Et = E_1 + E_2_index + En_3_index + En_4_index

        return Et

    def _Qibo_measure_Energy_fermionic_exact_fast(self, prefix_circuit=None):
        """
        Equivalent to Qibo_measure_Energy() for the fermionic, exact, default
        (circuits=None) case. Builds the standard ansatz prefix (ref state + layer,
        or just ref state if only_ref) unless a custom prefix_circuit is supplied
        (used by the fast gradient/GGF paths to evaluate the energy of an
        ansatz+offset/shift variant through the same cached-statevector mechanism),
        then delegates to _Qibo_energy_from_prefix.
        """
        if prefix_circuit is not None:
            main_circuit = prefix_circuit
        elif self.only_ref is False:
            main_circuit = self.Qibo_ref_state_composer() + self.Qibo_layer_composer()
        else:
            main_circuit = self.Qibo_ref_state_composer()

        return self._Qibo_energy_from_prefix(main_circuit)

    def Qibo_staircase_pauli_individual_exponential(self, pauli_op):

        circuits = []

        for term, coef in pauli_op.terms.items():
            circuit_1 = Circuit(self.n_qubits)
            if not term:
                continue

            # 1. Apply H or Rx gates in the X or Y Pauli exponentials qubits
            for q, p in term:
                if p == "X":
                    circuit_1.add(gates.H(q))
                elif p == "Y":
                    circuit_1.add(gates.RX(q, theta=np.pi / 2))

            # 2. Staircase algorithm
            qubit_indices = [q for q, _ in term]
            for i in range(len(qubit_indices) - 1):
                circuit_1.add(gates.CNOT(qubit_indices[i], qubit_indices[i + 1]))

            # 3. Apply a theta rotation
            last_qubit = qubit_indices[-1]
            circuit_1.add(gates.RZ(last_qubit, theta=np.pi / 2))

            # 4. Last part staircase structure
            for i in reversed(range(len(qubit_indices) - 1)):
                circuit_1.add(gates.CNOT(qubit_indices[i], qubit_indices[i + 1]))

            # 5. Apply H or Rx gates in the X or Y Pauli exponentials qubits
            for q, p in term:
                if p == "X":
                    circuit_1.add(gates.H(q))
                elif p == "Y":
                    circuit_1.add(gates.RX(q, theta=-np.pi / 2))
            ####################################################################################################################
            circuit_2 = Circuit(self.n_qubits)
            if not term:
                continue

            # 1. Apply H or Rx gates in the X or Y Pauli exponentials qubits
            for q, p in term:
                if p == "X":
                    circuit_2.add(gates.H(q))
                elif p == "Y":
                    circuit_2.add(gates.RX(q, theta=np.pi / 2))

            # 2. Staircase algorithm
            qubit_indices = [q for q, _ in term]
            for i in range(len(qubit_indices) - 1):
                circuit_2.add(gates.CNOT(qubit_indices[i], qubit_indices[i + 1]))

            # 3. Apply a theta rotation
            last_qubit = qubit_indices[-1]
            circuit_2.add(gates.RZ(last_qubit, theta=-np.pi / 2))

            # 4. Last part staircase structure
            for i in reversed(range(len(qubit_indices) - 1)):
                circuit_2.add(gates.CNOT(qubit_indices[i], qubit_indices[i + 1]))

            # 5. Apply H or Rx gates in the X or Y Pauli exponentials qubits
            for q, p in term:
                if p == "X":
                    circuit_2.add(gates.H(q))
                elif p == "Y":
                    circuit_2.add(gates.RX(q, theta=-np.pi / 2))

            circuits.append([circuit_1, circuit_2, coef.real])

        return circuits

    def Qibo_gradient_circuits(self, op_index):

        if self.encoding == "HCB":
            A, B = op_index

            main_circ = self.Qibo_ref_state_composer() + self.Qibo_layer_composer()

            circ_plus = main_circ.copy()
            circ_min = main_circ.copy()
            circ_plus_plus = main_circ.copy()
            circ_min_min = main_circ.copy()

            d = 0.4 * np.pi

            circ_plus.add(gates.GIVENS(A, B, -d))
            circ_min.add(gates.GIVENS(A, B, d))
            circ_plus_plus.add(gates.GIVENS(A, B, -2 * d))
            circ_min_min.add(gates.GIVENS(A, B, 2 * d))

            if self.complex:
                circs_plus = self.Qibo_all_circuits_complex(circ_plus)
                circs_min = self.Qibo_all_circuits_complex(circ_min)
                circs_plus_plus = self.Qibo_all_circuits_complex(circ_plus_plus)
                circs_min_min = self.Qibo_all_circuits_complex(circ_min_min)
            else:
                circs_plus = self.Qibo_all_circuits(circ_plus)
                circs_min = self.Qibo_all_circuits(circ_min)
                circs_plus_plus = self.Qibo_all_circuits(circ_plus_plus)
                circs_min_min = self.Qibo_all_circuits(circ_min_min)

            return circs_plus, circs_plus_plus, circs_min, circs_min_min

        gradient_circuits = []
        pauli_op = self.Operator_index_to_Pauli(op_index)
        gradient_apendix = self.Qibo_staircase_pauli_individual_exponential(
            pauli_op=pauli_op
        )

        for circ in gradient_apendix:
            pairs = []
            for n in range(2):
                if self.only_ref is False:
                    main_circuit = (
                        self.Qibo_ref_state_composer()
                        + self.Qibo_layer_composer()
                        + circ[n]
                    )
                else:
                    main_circuit = self.Qibo_ref_state_composer() + circ[n]

                circuits = [[main_circuit, 0, 0]]
                circuits_3 = {}

                for op in self.operator_pool:

                    indexes = op.ijkl
                    matrix_element = op.H2b

                    if len(set(indexes)) == 4:
                        circuit = main_circuit + self.qibo_4_dif_index(
                            operator_index=indexes
                        )
                        circuits.append([circuit, matrix_element, indexes])

                    elif len(set(indexes)) == 3:
                        counter = Counter(indexes)
                        repetido = [num for num, count in counter.items() if count > 1][
                            0
                        ]
                        j, k = [num for num in indexes if num != repetido]

                        jk_sorted = tuple(sorted([j, k]))

                        if jk_sorted in circuits_3:
                            circuits_3[jk_sorted][1].append(matrix_element)
                            circuits_3[jk_sorted][2].append(repetido)
                            circuits_3[jk_sorted][3].append(indexes)
                        else:
                            j, k = sorted([j, k])
                            circuit = main_circuit + self.qibo_3_dif_index(j=j, k=k)
                            circuits_3[jk_sorted] = [
                                circuit,
                                [matrix_element],
                                [repetido],
                                [indexes],
                                jk_sorted,
                            ]

                for key in circuits_3.keys():
                    circuits.append(circuits_3[key])

                pairs.append(circuits)
            pairs.append(circ[2])
            gradient_circuits.append(pairs)

        return gradient_circuits

    def obs_P(self, ijkl: list):

        obs = []
        i, j, k, l = self._local_list(ijkl)

        for m in range(i + 1, j):
            if m < k or m > l:
                obs.append(m)

        for n in range(k + 1, l):
            if n < i or n > j:
                obs.append(n)

        return obs

    def Qibo_measure_Energy(self, circuits=None):

        if self.encoding == "HCB":
            if self.complex:
                return self.Qibo_measure_Energy_complex(circuits)
            else:
                return self.Qibo_measure_Energy_simple(circuits)

        if circuits is None and self.exact and self.statevector_caching:
            return self._Qibo_measure_Energy_fermionic_exact_fast()

        if circuits == None:
            Qibo_circs = self.Qibo_all_circuits()
        else:
            Qibo_circs = circuits

        E_1 = 0
        E_2_index = 0
        main_circ = Qibo_circs[0][0].copy()

        if not self.exact:

            main_circ.add([gates.M(i) for i in range(main_circ.nqubits)])

            result = main_circ.execute(nshots=self.nshots)

            result = MeasurementOutcomes(
                measurements=result.measurements,
                backend=result.backend,
                probabilities=None,
                samples=result.samples(),
                nshots=self.nshots,
            )

        else:
            result = main_circ()
            print(result)

        print("Energy,\t\t\tobservable,\t\tamplitude\n")

        for part in self.monoparticular_energies:
            i = int(part)
            eps = self.monoparticular_energies[part]

            E_i = eps * result.probabilities([self._local(i)])[1]

            E_1 += E_i

            if abs(E_i) > 10 ** (-3):
                print(np.round(E_i, 8), "\t\t", i, "\t\t", np.round(eps, 8))

        for op in self.two_index:

            nu_ij = op.H2b
            ij = set(op.ijkl)

            E_i = nu_ij * result.probabilities(set(self._local_list(ij)))[3]
            E_2_index += E_i

            if abs(E_i) > 10 ** (-3):
                print(np.round(E_i, 8), "\t\t", op.ijkl, "\t\t", np.round(nu_ij, 8))

        En_4_index = 0
        En_3_index = 0

        for i in range(1, len(Qibo_circs)):
            circ = Qibo_circs[i][0].copy()

            if not self.exact:
                circ.add([gates.M(i) for i in range(circ.nqubits)])
                result = circ.execute(nshots=self.nshots)

                res = MeasurementOutcomes(
                    measurements=result.measurements,
                    backend=result.backend,
                    probabilities=None,
                    samples=result.samples(),
                    nshots=self.nshots,
                )
            else:
                res = circ()

            if (
                len(Qibo_circs[i]) == 3
            ):  # the data structure for the 4 diferent index just has 3 parameters

                op_amplitude = Qibo_circs[i][1]
                op_index = Qibo_circs[i][2]

                indices = self.obs_P(op_index)

                bitstrings = []
                for numero in range(2 ** len(indices)):
                    bitstring = bin(numero)[2:]
                    bitstrings.append(bitstring)

                ordenes = indices + self._local_list(op_index)

                E_ijkl = 0

                for bitstring in bitstrings:
                    n = 0
                    for bit in bitstring:
                        n += int(bit)

                    sign = (-1) ** n  # Alterna entre sumar y restar

                    E_ijkl += (
                        sign
                        * op_amplitude
                        * (
                            res.probabilities(ordenes)[int(bitstring + "1100", 2)]
                            - res.probabilities(ordenes)[int(bitstring + "0011", 2)]
                        )
                    )

                En_4_index += E_ijkl

                if abs(E_ijkl) > 10 ** (-3):
                    print(
                        np.round(E_ijkl, 8),
                        "\t\t",
                        op_index,
                        "\t\t",
                        np.round(op_amplitude, 8),
                    )

            elif len(Qibo_circs[i]) == 5:

                op_amplitudes = Qibo_circs[i][1]
                reps = Qibo_circs[i][2]
                j, k = Qibo_circs[i][4]
                op_indexs = Qibo_circs[i][3]

                for r in range(len(reps)):
                    rep = reps[r]
                    op_index = op_indexs[r]
                    indices = self.obs_P(op_index)

                    op_index = [rep, j, k]
                    op_amplitude = op_amplitudes[r]

                    bitstrings = []
                    for numero in range(2 ** len(indices)):
                        bitstring = bin(numero)[2:]
                        bitstrings.append(bitstring)

                    ordenes = indices + self._local_list(op_index)

                    E_njnk = 0

                    for bitstring in bitstrings:
                        par = 0
                        for bit in bitstring:
                            par += int(bit)

                        sign = (-1) ** par  # Alterna entre sumar y restar

                        E_njnk += (
                            sign
                            * op_amplitude
                            * (
                                res.probabilities(ordenes)[int(bitstring + "110", 2)]
                                - res.probabilities(ordenes)[int(bitstring + "101", 2)]
                            )
                        )

                    En_3_index += E_njnk

                    if abs(E_njnk) > 10 ** (-3):
                        print(
                            np.round(E_njnk, 8),
                            "\t\t",
                            op_index,
                            "\t\t",
                            np.round(op_amplitude, 8),
                        )

        print(
            "\n-----------------------------------Final Energies-----------------------------------\n"
        )
        print("Monoparticular energies:                 ", E_1)
        print("Two body energies with 2 diferent index: ", E_2_index)
        print("Two body energies with 3 diferent index: ", En_3_index)
        print("Two body energies with 4 diferent index: ", En_4_index)

        Et = E_1 + E_2_index + En_3_index + En_4_index

        return Et

    def Qibo_measure_gradient(self, op_index):

        if self.encoding == "HCB":
            A, B = op_index[0], op_index[1]

            C2 = (1 - np.sqrt(2)) / 2

            circs_plus, circs_plus_plus, circs_min, circs_min_min = (
                self.Qibo_gradient_circuits((A, B))
            )

            gradient = (
                self.Qibo_measure_Energy(circs_plus)[0]
                - self.Qibo_measure_Energy(circs_min)[0]
            ) + C2 * (
                self.Qibo_measure_Energy(circs_plus_plus)[0]
                - self.Qibo_measure_Energy(circs_min_min)[0]
            )

            print(gradient)

            return gradient

        if self.exact and self.statevector_caching:
            return self._Qibo_measure_gradient_fermionic_exact_fast(op_index)

        gradient = 0
        gradient_circs = self.Qibo_gradient_circuits(op_index)
        for circs in gradient_circs:
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                try:
                    en1 = self.Qibo_measure_Energy(circuits=circs[0])
                    en2 = self.Qibo_measure_Energy(circuits=circs[1])
                except:
                    en1 = 0
                    en2 = 0
            gradient = gradient + (en1 - en2) * circs[2]

        return gradient

    def _Qibo_measure_gradient_fermionic_exact_fast(self, op_index):
        """
        Equivalent to the fermionic branch of Qibo_measure_gradient() for the exact
        (noiseless) case, but evaluates each parameter-shift energy through
        _Qibo_energy_from_prefix (cached ansatz-prefix statevector reused across all
        Hamiltonian-term suffix circuits) instead of, for every Hamiltonian term,
        fully rebuilding and re-simulating a separate ansatz+shift+term circuit from
        scratch. The ref-state+layer prefix is also built only once here (it is the
        same for every Pauli term / shift sign) instead of once per (Pauli term x
        shift sign x Hamiltonian term) as the slow path effectively did.
        """
        pauli_op = self.Operator_index_to_Pauli(op_index)
        gradient_apendix = self.Qibo_staircase_pauli_individual_exponential(
            pauli_op=pauli_op
        )

        if self.only_ref is False:
            base_circuit = self.Qibo_ref_state_composer() + self.Qibo_layer_composer()
        else:
            base_circuit = self.Qibo_ref_state_composer()

        gradient = 0
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            for circ_plus_shift, circ_minus_shift, coef in gradient_apendix:
                try:
                    en1 = self._Qibo_energy_from_prefix(base_circuit + circ_plus_shift)
                    en2 = self._Qibo_energy_from_prefix(
                        base_circuit + circ_minus_shift
                    )
                except:
                    en1 = 0
                    en2 = 0
                gradient = gradient + (en1 - en2) * coef

        return gradient

    def Qibo_measure_energy_no_diagonalization_4_index(self, circuits=None):

        if circuits == None:
            Qibo_circs = self.circuits_no_4_index_diagonalization()
        else:
            Qibo_circs = circuits

        E_1 = 0
        E_2_index = 0
        main_circ = Qibo_circs[0][0].copy()

        if not self.exact:

            main_circ.add([gates.M(i) for i in range(main_circ.nqubits)])

            result = main_circ.execute(nshots=self.nshots)

            result = MeasurementOutcomes(
                measurements=result.measurements,
                backend=result.backend,
                probabilities=None,
                samples=result.samples(),
                nshots=self.nshots,
            )

        else:
            result = main_circ()
            print(result)

        print("Energy,\t\t\tobservable,\t\tamplitude\n")

        for part in self.monoparticular_energies:
            i = int(part)
            eps = self.monoparticular_energies[part]

            E_i = eps * result.probabilities([self._local(i)])[1]

            E_1 += E_i

            if abs(E_i) > 10 ** (-3):
                print(np.round(E_i, 8), "\t\t", i, "\t\t", np.round(eps, 8))

        for op in self.two_index:

            nu_ij = op.H2b
            ij = set(op.ijkl)

            E_i = nu_ij * result.probabilities(set(self._local_list(ij)))[3]
            E_2_index += E_i

            if abs(E_i) > 10 ** (-3):
                print(np.round(E_i, 8), "\t\t", op.ijkl, "\t\t", np.round(nu_ij, 8))

        En_4_index = 0
        En_3_index = 0

        E_index = defaultdict(float)

        for op in self.nuc.operators:
            E_index[tuple(op.ijkl)] = [0, op.H2b]

        for i in range(1, len(Qibo_circs)):
            circ = Qibo_circs[i][0].copy()

            if not self.exact:
                circ.add([gates.M(i) for i in range(circ.nqubits)])
                result = circ.execute(nshots=self.nshots)

                res = MeasurementOutcomes(
                    measurements=result.measurements,
                    backend=result.backend,
                    probabilities=None,
                    samples=result.samples(),
                    nshots=self.nshots,
                )
            else:
                res = circ()

            if (
                len(Qibo_circs[i]) == 4
            ):  # the data structure for the 4 diferent index just has 3 parameters

                op_amplitude = Qibo_circs[i][1]
                afected = Qibo_circs[i][2]
                op_index = tuple(Qibo_circs[i][3])
                bitstrings = []
                for numero in range(2 ** len(afected)):
                    bitstring = bin(numero)[2:]
                    bitstrings.append(bitstring)

                E_ijkl = 0

                for bitstring in bitstrings:

                    n = 0
                    for bit in bitstring:
                        n += int(bit)

                    sign = (-1) ** n

                    E_ijkl += (
                        sign
                        * op_amplitude
                        * (res.probabilities(afected)[int(bitstring, 2)])
                    )

                En_4_index += E_ijkl
                E_index[op_index][0] += E_ijkl

            elif len(Qibo_circs[i]) == 5:

                op_amplitudes = Qibo_circs[i][1]
                reps = Qibo_circs[i][2]
                j, k = Qibo_circs[i][4]
                op_indexs = Qibo_circs[i][3]

                for r in range(len(reps)):
                    rep = reps[r]
                    op_index = op_indexs[r]
                    indices = self.obs_P(op_index)

                    op_index = [rep, j, k]
                    op_amplitude = op_amplitudes[r]

                    bitstrings = []
                    for numero in range(2 ** len(indices)):
                        bitstring = bin(numero)[2:]
                        bitstrings.append(bitstring)

                    ordenes = indices + self._local_list(op_index)

                    E_njnk = 0

                    for bitstring in bitstrings:
                        par = 0
                        for bit in bitstring:
                            par += int(bit)

                        sign = (-1) ** par  # Alterna entre sumar y restar

                        E_njnk += (
                            sign
                            * op_amplitude
                            * (
                                res.probabilities(ordenes)[int(bitstring + "110", 2)]
                                - res.probabilities(ordenes)[int(bitstring + "101", 2)]
                            )
                        )

                    En_3_index += E_njnk

                    if abs(E_njnk) > 10 ** (-3):
                        print(
                            np.round(E_njnk, 8),
                            "\t\t",
                            op_index,
                            "\t\t",
                            np.round(op_amplitude, 8),
                        )

        for index in E_index:
            if abs(E_index[index][0]) > 10 ** (-8):
                print(
                    np.round(E_index[index][0], 8),
                    "\t\t",
                    list(index),
                    "\t\t",
                    np.round(E_index[index][1], 8),
                )

        print(
            "\n-----------------------------------Final Energies-----------------------------------\n"
        )
        print("Monoparticular energies:                 ", E_1)
        print("Two body energies with 2 diferent index: ", E_2_index)
        print("Two body energies with 3 diferent index: ", En_3_index)
        print("Two body energies with 4 diferent index: ", En_4_index)

        Et = E_1 + E_2_index + En_3_index + En_4_index

        return Et

    def circuits_no_4_index_diagonalization(self):

        if self.only_ref is False:
            main_circuit = self.Qibo_ref_state_composer() + self.Qibo_layer_composer()
        else:
            main_circuit = self.Qibo_ref_state_composer()

        circuits = [[main_circuit, 0, 0]]
        circuits_3 = {}

        for op in self.operator_pool:

            indexes = op.ijkl
            matrix_element = op.H2b

            if len(set(indexes)) == 4:
                observable = self.Observable_index_to_Pauli(indexes)

                for term, coef in observable.terms.items():
                    circuit = main_circuit.copy()
                    if not term:
                        continue
                    obs = []
                    for q, p in term:
                        obs.append(q)
                        if p == "X":
                            circuit.add(gates.H(q))
                        elif p == "Y":
                            circuit.add(gates.RX(q, theta=np.pi / 2))

                    circuits.append(
                        [circuit, -matrix_element * coef.real, obs, indexes]
                    )

            elif len(set(indexes)) == 3:
                counter = Counter(indexes)
                repetido = [num for num, count in counter.items() if count > 1][0]
                j, k = [num for num in indexes if num != repetido]

                jk_sorted = tuple(sorted([j, k]))

                if jk_sorted in circuits_3:
                    circuits_3[jk_sorted][1].append(matrix_element)
                    circuits_3[jk_sorted][2].append(repetido)
                    circuits_3[jk_sorted][3].append(indexes)
                else:
                    j, k = sorted([j, k])
                    circuit = main_circuit + self.qibo_3_dif_index(j=j, k=k)
                    circuits_3[jk_sorted] = [
                        circuit,
                        [matrix_element],
                        [repetido],
                        [indexes],
                        jk_sorted,
                    ]

        for key in circuits_3.keys():
            circuits.append(circuits_3[key])

        return circuits

    def Qibo_GGF_offset_circuits(self, op_index):
        """
        Builds the four circuits (current ansatz plus the candidate operator applied
        with parameter +d, -d, +2d and -2d, where d = 0.4*pi) used by the GGF analytic
        Fourier-landscape fit in Qibo_find_min, for either encoding.
        """
        if self.encoding == "HCB":
            return self.Qibo_gradient_circuits(op_index)

        d = 0.4 * np.pi
        base_circuit = self.Qibo_ref_state_composer() + self.Qibo_layer_composer()

        circ_plus = base_circuit + self._Qibo_staircase_single_exponential(op_index, d)
        circ_min = base_circuit + self._Qibo_staircase_single_exponential(op_index, -d)
        circ_plus_plus = base_circuit + self._Qibo_staircase_single_exponential(
            op_index, 2 * d
        )
        circ_min_min = base_circuit + self._Qibo_staircase_single_exponential(
            op_index, -2 * d
        )

        circs_plus = self.Qibo_all_circuits(circ_plus)
        circs_min = self.Qibo_all_circuits(circ_min)
        circs_plus_plus = self.Qibo_all_circuits(circ_plus_plus)
        circs_min_min = self.Qibo_all_circuits(circ_min_min)

        return circs_plus, circs_plus_plus, circs_min, circs_min_min

    def _Qibo_staircase_single_exponential(self, op_index, theta):
        """Builds the staircase circuit for a single fermionic excitation exp(i*theta*op)."""
        circuit = Circuit(self.n_qubits)
        pauli_op = self.Operator_index_to_Pauli(op_index)
        self.Qibo_staircase_pauli_exponential(circuit, pauli_op, theta)
        return circuit

    def _Qibo_scalar_energy(self, measurement):
        """Extracts the scalar energy from a Qibo_measure_Energy() result (HCB returns a tuple)."""
        if self.encoding == "HCB":
            return measurement[0]
        return measurement

    def Qibo_find_min(self, op_index):

        d = 0.4 * np.pi
        cd = np.cos(d)
        sd = np.sin(d)
        c2d = np.cos(2 * d)
        s2d = np.sin(2 * d)

        if self.encoding == "fermionic" and self.exact and self.statevector_caching:
            # Fast path: evaluate each of the 4 GGF parameter-offset energies through
            # _Qibo_energy_from_prefix, which simulates the (ansatz+offset) prefix
            # once and reuses its cached statevector for every Hamiltonian-term
            # suffix circuit, instead of building a full ansatz+offset+term circuit
            # per Hamiltonian term and re-simulating it from scratch (what
            # Qibo_GGF_offset_circuits + Qibo_measure_Energy(circuits=...) do below).
            base_circuit = self.Qibo_ref_state_composer() + self.Qibo_layer_composer()

            with contextlib.redirect_stdout(self.f):

                E0 = self.E0

                E_p2 = self._Qibo_energy_from_prefix(
                    base_circuit
                    + self._Qibo_staircase_single_exponential(op_index, 2 * d)
                )

                E_p1 = self._Qibo_energy_from_prefix(
                    base_circuit + self._Qibo_staircase_single_exponential(op_index, d)
                )

                E_m2 = self._Qibo_energy_from_prefix(
                    base_circuit
                    + self._Qibo_staircase_single_exponential(op_index, -2 * d)
                )

                E_m1 = self._Qibo_energy_from_prefix(
                    base_circuit
                    + self._Qibo_staircase_single_exponential(op_index, -d)
                )
        else:
            circs_plus, circs_plus_plus, circs_min, circs_min_min = (
                self.Qibo_GGF_offset_circuits(op_index)
            )

            with contextlib.redirect_stdout(self.f):

                E0 = self.E0

                E_p2 = self._Qibo_scalar_energy(self.Qibo_measure_Energy(circs_plus_plus))

                E_p1 = self._Qibo_scalar_energy(self.Qibo_measure_Energy(circs_plus))

                E_m2 = self._Qibo_scalar_energy(self.Qibo_measure_Energy(circs_min_min))

                E_m1 = self._Qibo_scalar_energy(self.Qibo_measure_Energy(circs_min))

        # E(t) = c0 + c1*cos(t) + s1*sin(t) + c2*cos(2t) + s2*sin(2t)

        sum_1 = E_p1 + E_m1
        dif_1 = E_p1 - E_m1
        sum_2 = E_p2 + E_m2
        dif_2 = E_p2 - E_m2

        # Fourier normalization factor (2/N)
        norm = 0.4

        # DC component (simple average)
        c0 = (E0 + sum_1 + sum_2) / 5.0

        # Fundamental frequency (w)
        c1 = norm * (E0 + sum_1 * cd + sum_2 * c2d)
        s1 = norm * (dif_1 * sd + dif_2 * s2d)

        # Second frequency (2w)
        # Note: cos(4d) = cos(d) and sin(4d) = -sin(d)
        c2 = norm * (E0 + sum_1 * c2d + sum_2 * cd)
        s2 = norm * (dif_1 * s2d - dif_2 * sd)

        def objective_function(theta_array):
            theta = theta_array[0]  # L-BFGS expects arrays

            # Precompute trig terms for efficiency
            ct = np.cos(theta)
            st = np.sin(theta)
            c2t = np.cos(2 * theta)  # could also use 2*ct*ct - 1
            s2t = np.sin(2 * theta)  # could also use 2*st*ct

            # Energy value
            E_val = c0 + c1 * ct + s1 * st + c2 * c2t + s2 * s2t

            # Gradient value (Jacobian)
            # d/dt (cos t) = -sin t, etc.
            grad_val = -c1 * st + s1 * ct - 2 * c2 * s2t + 2 * s2 * c2t

            return E_val, np.array([grad_val])

        amp_fund = np.sqrt(c1**2 + s1**2)  # Amplitude of the pi frequency
        amp_2nd = np.sqrt(c2**2 + s2**2)  # Amplitude of the 2pi frequency

        # 2. Noise tolerance (e.g. 0.1 means 10%): if amp_fund is below 10% of
        # amp_2nd, treat it as noise/irrelevant.
        epsilon = 0.2

        if amp_fund < epsilon * amp_2nd or len(self.operators_used) == 0:
            grid = np.linspace(-np.pi / 2, np.pi / 2, 20)
        else:
            grid = np.linspace(-np.pi, np.pi, 20)

        # Only evaluate the energy on the grid (vectorized, very fast)

        vals = (
            c0
            + c1 * np.cos(grid)
            + s1 * np.sin(grid)
            + c2 * np.cos(2 * grid)
            + s2 * np.sin(2 * grid)
        )

        initial_guess = [grid[np.argmin(vals)]]

        # L-BFGS-B optimization with Jacobian
        # bounds=[-pi, pi] keeps the search bounded
        res = minimize(
            objective_function,
            initial_guess,
            method="L-BFGS-B",
            jac=True,  # Indicates that objective_function returns (val, grad)
            bounds=[(-np.pi, np.pi)],
            tol=1e-12,
        )

        return res.x[0], res.fun

    def Qibo_all_circuits_complex(self, circ=None):

        if circ == None:
            main_circuit = self.Qibo_ref_state_composer() + self.Qibo_layer_composer()
        else:
            main_circuit = circ

        if self.exact and self.statevector_caching:
            # Fast-path marker (consumed by Qibo_measure_Energy_complex): skip
            # pre-building one FULL ansatz+H(A)+H(B) circuit per hopping pair (and
            # the ansatz+H-on-every-qubit circuit) here. main_circuit only needs to
            # be simulated once; _Qibo_energy_complex_from_prefix reuses its cached
            # statevector as initial_state= for a small per-pair suffix circuit
            # instead of re-simulating the whole ansatz from scratch for every one
            # of the O(n_hcb^2) hopping pairs.
            return [main_circuit, None, None]

        op_circuits = []

        if self.hop_measurement == "givens":
            # Diagonalizes XX+YY into Zp-Zq per pair (paper Eq. 17): apply the
            # inverse Givens rotation U^dagger(pi/4) = GIVENS(-pi/4) to each pair
            # before measuring in the computational basis. As with the Hadamard
            # circuits below, this is only used downstream to pin down the
            # *sign* of the term (the magnitude is reconstructed separately from
            # the unrotated Z-basis coherence) - but unlike H(A);H(B), this
            # rotation conserves particle number on the pair. Unlike the
            # Hadamard case, there is no single global circuit that diagonalizes
            # every pair at once (pairs can share qubits), so no global circ_x
            # is built here.
            for op in self.ops_hop:
                circuit = main_circuit.copy()
                circuit.add(gates.GIVENS(op.A, op.B, -np.pi / 4))

                op_circuits.append((circuit, op))

            return [main_circuit, op_circuits, None]

        for op in self.ops_hop:
            circuit = main_circuit.copy()
            circuit.add(gates.H(op.A))
            circuit.add(gates.H(op.B))

            op_circuits.append((circuit, op))

        circ_x = main_circuit.copy()

        for q in range(self.n_qubits):
            circ_x.add(gates.H(q))

        return [main_circuit, op_circuits, circ_x]

    def Qibo_measure_Energy_simple(self, circuits=None):

        if circuits == None:
            Qibo_circs = self.Qibo_all_circuits()
        else:
            Qibo_circs = circuits

        E_diag = 0

        E_hop = 0

        E_hop_rec = 0

        main_circ = Qibo_circs[0].copy()
        circ_x = Qibo_circs[1].copy()
        # circ_y = Qibo_circs[2].copy()

        if not self.exact:

            main_circ.add([gates.M(i) for i in range(main_circ.nqubits)])
            circ_x.add([gates.M(i) for i in range(main_circ.nqubits)])
            # circ_y.add([gates.M(i) for i in range(main_circ.nqubits)])

            result = main_circ.execute(nshots=self.nshots)
            result_X = circ_x.execute(nshots=self.nshots)
            # result_Y = circ_y.execute(nshots=self.nshots)

            result = MeasurementOutcomes(
                measurements=result.measurements,
                backend=result.backend,
                probabilities=None,
                samples=result.samples(),
                nshots=self.nshots,
            )
            result_X = MeasurementOutcomes(
                measurements=result.measurements,
                backend=result.backend,
                probabilities=None,
                samples=result_X.samples(),
                nshots=self.nshots,
            )
            # result_Y = MeasurementOutcomes(measurements=result.measurements, backend=result.backend, probabilities=None, samples=result_Y.samples(), nshots=self.nshots)

        else:

            result = main_circ()
            result_X = circ_x()
            # result_Y = circ_y()
            print(result)

        for op in self.ops_1_body:
            E_diag += op.g * result.probabilities([op.A])[1]

        for op in self.ops_2_body:
            E_diag += op.g * result.probabilities([op.A, op.B])[3]

        for op in self.ops_hop:

            probs_z = result.probabilities([op.A, op.B])
            P_10 = probs_z[2]
            P_01 = probs_z[1]

            magnitude_clean = 2 * np.sqrt(P_10 * P_01)

            prob_x = result_X.probabilities([op.A, op.B])
            val_xx = prob_x[0] - prob_x[1] - prob_x[2] + prob_x[3]

            # prob_y = result_Y.probabilities([op.A, op.B])
            # val_yy = prob_y[0] - prob_y[1] - prob_y[2] + prob_y[3]

            signal = val_xx  # + val_yy

            if abs(signal) == 0:
                term_energy = 0.0
            else:
                sign_clean = np.sign(signal)

                term_energy = op.g * sign_clean * magnitude_clean

            E_hop_rec += term_energy

            E_hop += op.g * signal

        E_t = E_diag + E_hop_rec

        return E_t, E_diag, E_hop, result

    def Qibo_measure_Energy_complex(self, circuits=None):

        if circuits == None:
            Qibo_circs = self.Qibo_all_circuits_complex()
        else:
            Qibo_circs = circuits

        if self.exact and Qibo_circs[1] is None:
            # Fast-path marker from Qibo_all_circuits_complex: Qibo_circs[0] is just
            # the ansatz/offset prefix, with no per-pair circuits pre-built.
            return self._Qibo_energy_complex_from_prefix(Qibo_circs[0])

        E_diag = 0

        E_hop = 0

        E_hop_rec = 0

        results_x = []
        main_circ = Qibo_circs[0].copy()
        circs_x = Qibo_circs[1].copy()
        circ_global_x = Qibo_circs[2].copy() if Qibo_circs[2] is not None else None

        if not self.exact:

            main_circ.add([gates.M(i) for i in range(main_circ.nqubits)])
            result = main_circ.execute(nshots=self.nshots)
            result = MeasurementOutcomes(
                measurements=result.measurements,
                backend=result.backend,
                probabilities=None,
                samples=result.samples(),
                nshots=self.nshots,
            )

            if circ_global_x is not None:
                circ_global_x.add([gates.M(i) for i in range(main_circ.nqubits)])
                result_global_x = circ_global_x.execute(nshots=self.nshots)
                result_global_x = MeasurementOutcomes(
                    measurements=result_global_x.measurements,
                    backend=result.backend,
                    probabilities=None,
                    samples=result_global_x.samples(),
                    nshots=self.nshots,
                )

            for circ, op in circs_x:
                circ_x = circ.copy()
                circ_x.add([gates.M(i) for i in range(main_circ.nqubits)])
                result_X = circ_x.execute(nshots=self.nshots)
                result_X = MeasurementOutcomes(
                    measurements=result.measurements,
                    backend=result.backend,
                    probabilities=None,
                    samples=result_X.samples(),
                    nshots=self.nshots,
                )
                results_x.append((result_X, op))

        else:

            result = main_circ()
            result_global_x = circ_global_x() if circ_global_x is not None else None

            for circ_x, op in circs_x:
                result_X = circ_x()
                results_x.append((result_X, op))

            print(result)

        for op in self.ops_1_body:
            E_diag += op.g * result.probabilities([op.A])[1]

        for op in self.ops_2_body:
            E_diag += op.g * result.probabilities([op.A, op.B])[3]

        for res, op in results_x:

            indexes_k = [q for q in range(self.n_qubits) if q != op.A and q != op.B]
            indexes = indexes_k + [op.A, op.B]

            traced_probs = result.probabilities([op.A, op.B])

            if traced_probs[1] * traced_probs[2] != 0:

                # The magnitude is always reconstructed from the *unrotated*
                # Z-basis coherence - this is exact for these real-amplitude
                # states and identical for both hop_measurement strategies. Only
                # the rotation used to pin down the *sign* differs: "givens"
                # rotates with GIVENS(-pi/4) (diagonalizing XX+YY into Zp-Zq,
                # paper Eq. 17, and - unlike Hadamard - conserving particle
                # number on this pair), "hadamard" rotates with H(A);H(B).
                probs_z = result.probabilities(indexes)
                prob_x = res.probabilities(indexes)

                pz_mat = probs_z.reshape(-1, 4)
                px_mat = prob_x.reshape(-1, 4)

                magnitudes = 2 * np.sqrt(pz_mat[:, 1] * pz_mat[:, 2])

                if self.hop_measurement == "givens":
                    signals = px_mat[:, 1] - px_mat[:, 2]
                else:
                    signals = px_mat[:, 0] - px_mat[:, 1] - px_mat[:, 2] + px_mat[:, 3]

                signs = np.sign(signals)
                term_energies = op.g * signs * magnitudes
                E_hop_rec += np.sum(term_energies)

        if self.hop_measurement == "givens":
            # No single circuit can diagonalize every hopping pair at once with
            # Givens rotations when pairs share qubits (unlike the single global
            # H-on-every-qubit circuit used below for Hadamard), so there's no
            # analogous cheap global estimate here - reuse the per-pair
            # reconstruction above as E_hop too.
            E_hop = E_hop_rec
        else:
            for op in self.ops_hop:

                prob_x = result_global_x.probabilities([op.A, op.B])
                val_xx = prob_x[0] - prob_x[1] - prob_x[2] + prob_x[3]

                signal = val_xx

                E_hop += op.g * signal

        E_t = E_diag + E_hop_rec

        return E_t, E_diag, E_hop, result

    def _Qibo_energy_complex_from_prefix(self, main_circuit):
        """
        Equivalent to Qibo_measure_Energy_complex() for the HCB "complex" reference
        state, exact (noiseless) case, but simulates main_circuit (the ansatz, or an
        ansatz+GIVENS-offset/shift variant used by the gradient/GGF paths) exactly
        once and reuses its cached statevector as initial_state= for every hopping
        pair's small basis-rotation suffix circuit (GIVENS(-pi/4), or H(A)+H(B) for
        the "hadamard" method), instead of rebuilding and re-simulating a full
        ansatz+suffix circuit from scratch for every one of the O(n_hcb^2) pairs.

        For both methods the magnitude is reconstructed from the unrotated Z-basis
        coherence (probs_z, computed once from the cached statevector and reused for
        every pair via a plain numpy transpose, instead of issuing a fresh qibo
        probabilities() call - which redoes abs(state)**2 from scratch - for every
        pair; this is an exact replacement, not an approximation: qibo's own
        NumpyBackend.calculate_probabilities()/._order_probabilities() reduce to
        exactly this transpose when every qubit is included in the requested list,
        i.e. no marginalization, which is always the case here since `indexes` below
        spans all n_qubits). Only the rotation used to determine the term's *sign*
        differs between the two methods (see below).
        """
        result = main_circuit()
        print(result)
        main_state = np.array(result.state())

        # Full-system probabilities in natural (qubit-index-ascending) order,
        # computed once; each pair below just transposes this into its own
        # [indexes_k..., A, B] order instead of recomputing abs(state)**2 from qibo.
        probs_natural = (np.abs(main_state) ** 2).reshape((2,) * self.n_qubits)

        if self.hop_measurement != "givens":
            circ_global_x = Circuit(self.n_qubits)
            for q in range(self.n_qubits):
                circ_global_x.add(gates.H(q))
            result_global_x = circ_global_x(initial_state=main_state.copy())

        E_diag = 0
        E_hop = 0
        E_hop_rec = 0

        for op in self.ops_1_body:
            E_diag += op.g * result.probabilities([op.A])[1]

        for op in self.ops_2_body:
            E_diag += op.g * result.probabilities([op.A, op.B])[3]

        for op in self.ops_hop:

            traced_probs = result.probabilities([op.A, op.B])

            if traced_probs[1] * traced_probs[2] != 0:

                indexes_k = [
                    q for q in range(self.n_qubits) if q != op.A and q != op.B
                ]
                indexes = indexes_k + [op.A, op.B]

                # The magnitude is always reconstructed from the unrotated
                # Z-basis coherence - identical for both hop_measurement
                # strategies. Only the rotation used to pin down the *sign*
                # differs: "givens" rotates with GIVENS(-pi/4) (diagonalizing
                # XX+YY into Zp-Zq, paper Eq. 17, and - unlike Hadamard -
                # conserving particle number on this pair), "hadamard" rotates
                # with H(A);H(B).
                suffix = Circuit(self.n_qubits)
                if self.hop_measurement == "givens":
                    suffix.add(gates.GIVENS(op.A, op.B, -np.pi / 4))
                else:
                    suffix.add(gates.H(op.A))
                    suffix.add(gates.H(op.B))

                res = suffix(initial_state=main_state.copy())
                rotated_state = np.array(res.state())
                probs_x_natural = (np.abs(rotated_state) ** 2).reshape(
                    (2,) * self.n_qubits
                )

                probs_z = np.transpose(probs_natural, axes=indexes).ravel()
                prob_x = np.transpose(probs_x_natural, axes=indexes).ravel()

                pz_mat = probs_z.reshape(-1, 4)
                px_mat = prob_x.reshape(-1, 4)

                magnitudes = 2 * np.sqrt(pz_mat[:, 1] * pz_mat[:, 2])

                if self.hop_measurement == "givens":
                    signals = px_mat[:, 1] - px_mat[:, 2]
                else:
                    signals = px_mat[:, 0] - px_mat[:, 1] - px_mat[:, 2] + px_mat[:, 3]

                signs = np.sign(signals)
                term_energies = op.g * signs * magnitudes
                E_hop_rec += np.sum(term_energies)

        if self.hop_measurement == "givens":
            # No single circuit can diagonalize every hopping pair at once with
            # Givens rotations when pairs share qubits (unlike the single global
            # H-on-every-qubit circuit used below for Hadamard), so there's no
            # analogous cheap global estimate here - reuse the per-pair
            # reconstruction above as E_hop too.
            E_hop = E_hop_rec
        else:
            for op in self.ops_hop:

                prob_x = result_global_x.probabilities([op.A, op.B])
                val_xx = prob_x[0] - prob_x[1] - prob_x[2] + prob_x[3]

                E_hop += op.g * val_xx

        E_t = E_diag + E_hop_rec

        return E_t, E_diag, E_hop, result

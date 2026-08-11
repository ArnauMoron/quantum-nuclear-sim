import numpy as np

from openfermion.ops import FermionOperator, QubitOperator
from openfermion.transforms import jordan_wigner

from qibo import gates
from qibo.models.circuit import Circuit

from CORE.Nucleus import Nucleus


class CircuitBuilder:
    """
    Shared, encoding-aware circuit-building infrastructure for either the
    fermionic or the HCB (hard-core boson) encoding: reference-state circuit
    construction, raw-to-local qubit re-basing, the staircase Pauli-exponential
    circuit builder, and Jordan-Wigner conversions of Hamiltonian terms (the
    Hermitian two-body piece and the one-body number operator).

    This holds nothing specific to either the ADAPT-VQE algorithm (operator
    pools, ansatz layers, gradient/GGF measurement - see adapt_vqe/Circuit.py's
    Circuits_Composer, which subclasses this) or the annealing algorithm (see
    annealing/Annealing.py's AnnealingProtocol, which also subclasses this) -
    both depend on it, neither depends on the other.
    """

    def __init__(
        self, nucleus: Nucleus, ref_state: int = 0, encoding: str = "fermionic"
    ) -> None:
        self.nuc = nucleus
        self.ref_state = ref_state
        self.encoding = encoding

        # Jordan-Wigner Pauli decompositions only depend on the fixed index
        # tuple and the static n_qubits/qubit_offset, so they are safe to cache
        # for the lifetime of this instance.
        self._observable_pauli_cache = {}
        self._onebody_pauli_cache = {}

        if encoding == "fermionic":
            self.n_qubits = len(nucleus.qubits)
            # Raw single-particle indices (from the data files) are not always
            # 0-indexed (e.g. a neutron-only sd-shell space starts at 12, not 0),
            # but Qibo circuits need contiguous 0-indexed qubit positions. This
            # offset re-bases every raw index to a local circuit qubit position,
            # mirroring how the HCB encoding already builds its own local (0-indexed)
            # quasiparticle mapping regardless of the raw indices involved.
            self.qubit_offset = nucleus.qubits[0]
        elif encoding == "HCB":
            self.ref_state_indexes = nucleus.hcb_states[ref_state]
            self.n_qubits = len(nucleus.nucleus.qubits) // 2

    def _local(self, idx):
        """Maps a raw (fermionic) single-particle index to its local, 0-indexed circuit qubit position."""
        return idx - self.qubit_offset

    def _local_list(self, indices):
        """Maps an iterable of raw single-particle indices to local circuit qubit positions."""
        return [self._local(idx) for idx in indices]

    def Qibo_ref_state_composer(self):

        if self.encoding == "HCB":
            circuit = Circuit(self.n_qubits)

            for index in self.ref_state_indexes:
                circuit.add(gates.X(index))

            return circuit

        # self.nuc.states already holds each many-body basis state as a tuple of
        # occupied single-particle indices (parsed once at Nucleus construction
        # time from the same mb_basis_2.dat file), so there is no need to re-open
        # and re-parse the file here.
        ref_state_index = self._local_list(self.nuc.states[self.ref_state])

        circuit = Circuit(self.n_qubits)

        for index in ref_state_index:
            circuit.add(gates.X(index))

        for q in range(self.n_qubits):
            if q not in ref_state_index:
                circuit.add(gates.I(q))

        return circuit

    def Observable_index_to_Pauli(self, index):
        """Converts the Hermitian two-body Hamiltonian piece a†_i a†_j a_l a_k
        (+ its Hermitian conjugate, unless (i,j)==(k,l) in which case the single
        term is already Hermitian on its own) into a Pauli string over n_qubits.

        This exact operator ordering/symmetrization (annihilation operators in
        [l, k] order, not the naively-expected [k, l]; no h.c. added when
        (i,j)==(k,l)) is not a free choice - it is what makes this match
        Nucleus.Ham_2_body_contributions()'s TwoBodyExcitationOperator.matrix for
        the same `index`, which is itself validated (see
        Nucleus.Ham_1_body_contributions/Ham_2_body_contributions's docstrings and
        the annealing module's consistency check) to reconstruct nucleus.H
        exactly. Verified by construction-time comparison against every
        TwoBodyExcitationOperator.matrix for a representative nucleus.

        Memoized per index: this can be called repeatedly (e.g. once per ADAPT
        circuit build, or once per annealing Hamiltonian construction) for a
        result that never changes.
        """
        key = tuple(index)
        cached = self._observable_pauli_cache.get(key)
        if cached is not None:
            return cached

        i, j, k, l = self._local_list(index)

        fermion_op = FermionOperator(f"{i}^ {j}^ {l} {k}", 1)  # a†_i a†_j a_l a_k
        if (i, j) != (k, l):
            fermion_op += FermionOperator(f"{k}^ {l}^ {j} {i}", 1)  # + h.c.

        pauli_op = jordan_wigner(fermion_op)

        filtered_pauli_op = QubitOperator()
        for term, coef in pauli_op.terms.items():
            if all(q < self.n_qubits for q, _ in term):
                filtered_pauli_op += QubitOperator(term, coef)

        self._observable_pauli_cache[key] = filtered_pauli_op
        return filtered_pauli_op

    def OneBody_index_to_Pauli(self, index):
        """Converts the one-body number operator a†_i a_i into a Pauli string over
        n_qubits (same pattern as Observable_index_to_Pauli, for the 1-body piece
        of the Hamiltonian - used by e.g. the annealing module's driver/target
        Hamiltonian construction)."""
        key = index
        cached = self._onebody_pauli_cache.get(key)
        if cached is not None:
            return cached

        i = self._local(index)

        fermion_op = FermionOperator(f"{i}^ {i}", 1)  # a†_i a_i

        pauli_op = jordan_wigner(fermion_op)

        filtered_pauli_op = QubitOperator()
        for term, coef in pauli_op.terms.items():
            if all(q < self.n_qubits for q, _ in term):
                filtered_pauli_op += QubitOperator(term, coef)

        self._onebody_pauli_cache[key] = filtered_pauli_op
        return filtered_pauli_op

    def Qibo_staircase_pauli_exponential(self, circuit, pauli_op, theta):
        """
        Builds a circuti of e^(i * theta * pauli_op) using the staircase algorithm.

        Parameters:
        - pauli_op: QubitOperator constructed with Pauli_ops
        - n_qubits
        - theta

        Returns:
        - The staircase circuit.
        """

        for term, coef in pauli_op.terms.items():
            if not term:
                continue

            # 1. Apply H or Rx gates in the X or Y Pauli exponentials qubits
            for q, p in term:
                if p == "X":
                    circuit.add(gates.H(q))
                elif p == "Y":
                    circuit.add(gates.RX(q, theta=np.pi / 2))

            # 2. Staircase algorithm
            qubit_indices = [q for q, _ in term]
            for i in range(len(qubit_indices) - 1):
                circuit.add(gates.CNOT(qubit_indices[i], qubit_indices[i + 1]))

            # 3. Apply a theta rotation
            last_qubit = qubit_indices[-1]
            circuit.add(gates.RZ(last_qubit, theta=2 * theta * coef.real))

            # 4. Last part staircase structure
            for i in reversed(range(len(qubit_indices) - 1)):
                circuit.add(gates.CNOT(qubit_indices[i], qubit_indices[i + 1]))

            # 5. Apply H or Rx gates in the X or Y Pauli exponentials qubits
            for q, p in term:
                if p == "X":
                    circuit.add(gates.H(q))
                elif p == "Y":
                    circuit.add(gates.RX(q, theta=-np.pi / 2))

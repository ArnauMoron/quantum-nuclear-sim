import numpy as np
from numpy import linalg as la
from scipy.linalg import expm

from openfermion.ops import QubitOperator
from openfermion.linalg import get_sparse_operator

from qibo import gates

from CORE.Nucleus import Nucleus
from CORE.CircuitBuilder import CircuitBuilder


def _linear_schedule(step: int, steps: int) -> float:
    """Default annealing schedule: s goes linearly from 0 (step=0) to 1 (step=steps)."""
    return step / steps


class AnnealingProtocol(CircuitBuilder):
    """
    Builds and simulates a time-Trotterized digital quantum annealing protocol,
    for either the fermionic or the HCB (hard-core boson) encoding.

    The protocol interpolates from an easily-preparable "driver" Hamiltonian H_D
    (a sum of single-orbital/mode number operators whose unique ground state, in
    the fixed-particle-number sector, is the chosen many-body reference state) to
    the real nuclear Hamiltonian H_T, and Trotter-simulates
    H(s) = (1 - s) * H_D + s * H_T along a schedule s: step -> [0, 1].

    Subclasses CORE.CircuitBuilder for the reference-state circuit, Jordan-Wigner
    conversions, qubit re-basing and staircase Pauli-exponential builder it needs
    - the same base class ADAPT-VQE's own Circuits_Composer (adapt_vqe/Circuit.py)
    subclasses, so this has no dependency on ADAPT-VQE's own (much heavier)
    operator-pool/gradient/GGF measurement machinery.

    Attributes:
        nucleus (Nucleus): The nucleus being annealed to.
        ref_state (int): Index (into nucleus.states/hcb_states) of the many-body
            basis state used as both H_D's ground state and the circuit's initial
            state.
        H_D_matrix, H_T_matrix (np.ndarray): Driver/target Hamiltonians in the
            small many-body basis (same basis as nucleus.H), used for the
            classical `exact_evolution` reference calculation.
        H_D_qubit, H_T_qubit (QubitOperator): Driver/target Hamiltonians as sums
            of Pauli strings, used to build the Trotterized circuit.
    """

    def __init__(
        self, nucleus: Nucleus, ref_state: int = 0, encoding: str = "fermionic"
    ) -> None:
        if nucleus.encoding != encoding:
            raise ValueError(
                f"Nucleus was built with encoding={nucleus.encoding!r}, expected "
                f"encoding={encoding!r}"
            )

        self.nucleus = nucleus

        super().__init__(nucleus, ref_state=ref_state, encoding=encoding)

        self._occupied = self._occupied_indices()
        self.n_occ = len(self._occupied)
        if self.n_occ == 0:
            raise ValueError(
                "ref_state has no occupied orbitals/modes; the driver Hamiltonian "
                "(which distributes -|E_ref| evenly across occupied "
                "orbitals/modes) needs at least one."
            )

        # ref_state is itself a many-body basis index, so <ref|H|ref> is a direct
        # diagonal lookup, not a matrix-vector product.
        self.E_ref = float(nucleus.H[ref_state, ref_state])

        self.H_D_matrix, self.H_T_matrix = self._build_matrices()
        self.H_D_qubit, self.H_T_qubit = self._build_qubit_hamiltonians()

    # ------------------------------------------------------------------
    # Hamiltonian construction
    # ------------------------------------------------------------------

    def _occupied_indices(self) -> list:
        """Orbitals (fermionic) or modes (HCB) occupied in ref_state."""
        if self.encoding == "fermionic":
            return list(self.nucleus.states[self.ref_state])
        return list(self.nucleus.hcb_states[self.ref_state])

    def _build_matrices(self) -> tuple:
        """
        Small (d_H- or d_hcb-dimensional) matrix forms of H_D and H_T, in the same
        many-body basis as nucleus.H. Used for the classical exact/instantaneous-
        eigenvalue reference calculation in exact_evolution().
        """
        H_T = self.nucleus.H

        if self.encoding == "fermionic":
            one_body_ops, _ = self.nucleus.Ham_1_body_contributions()
            by_index = {op.i: op for op in one_body_ops}
        else:
            by_index = {op.A: op for op in self.nucleus.ops_1_body}

        H_D = np.zeros_like(H_T)
        for idx in self._occupied:
            op = by_index.get(idx)
            if op is not None:
                H_D = H_D + op.matrix

        H_D = -abs(self.E_ref) / self.n_occ * H_D
        return H_D, H_T

    def _build_qubit_hamiltonians(self) -> tuple:
        """
        QubitOperator (openfermion) forms of H_D and H_T, used to build the
        Trotterized circuit (and, via get_sparse_operator, for the full-qubit-
        space energy/consistency checks).
        """
        if self.encoding == "fermionic":
            H_D = QubitOperator()
            for idx in self._occupied:
                H_D += self.OneBody_index_to_Pauli(idx)
            H_D = -abs(self.E_ref) / self.n_occ * H_D

            H_T = QubitOperator()
            for op in self.nucleus.Ham_2_body_contributions():
                H_T += self.Observable_index_to_Pauli(op.ijkl) * op.H2b
            one_body_ops, _ = self.nucleus.Ham_1_body_contributions()
            for op in one_body_ops:
                H_T += self.OneBody_index_to_Pauli(op.i) * op.eps

            return H_D, H_T

        # HCB: already qubit-native (each mode IS a qubit), no Jordan-Wigner
        # needed. N_A = (I - Z_A) / 2 is the number operator for mode A.
        H_D = QubitOperator()
        for A in self._occupied:
            H_D += QubitOperator("", 0.5) + QubitOperator(f"Z{A}", -0.5)
        H_D = -abs(self.E_ref) / self.n_occ * H_D

        H_T = QubitOperator()
        for op in self.nucleus.ops_1_body:
            H_T += op.g * (QubitOperator("", 0.5) + QubitOperator(f"Z{op.A}", -0.5))
        for op in self.nucleus.ops_2_body:
            n_a = QubitOperator("", 0.5) + QubitOperator(f"Z{op.A}", -0.5)
            n_b = QubitOperator("", 0.5) + QubitOperator(f"Z{op.B}", -0.5)
            H_T += op.g * (n_a * n_b)
        for op in self.nucleus.ops_hop:
            H_T += (
                0.5
                * op.g
                * (
                    QubitOperator(f"X{op.A} X{op.B}")
                    + QubitOperator(f"Y{op.A} Y{op.B}")
                )
            )

        return H_D, H_T

    def _hcb_coefficients(self) -> tuple:
        """
        Per-mode driver/target Z-coefficients and per-pair hopping/diagonal
        coefficients for the HCB encoding, shared by build_circuit (direct-gate)
        and build_swap_network_circuit. These are the "g" couplings themselves
        (H_T = sum g_A N_A + sum g_AB N_A N_B + sum 0.5 g_AB (X_A X_B + Y_A Y_B)),
        not yet turned into gate angles - see _rz_theta/_cu1_theta/_rxxyy_theta.
        """
        n = self.n_qubits
        h_driver = np.zeros(n)
        for A in self._occupied:
            h_driver[A] = -abs(self.E_ref) / self.n_occ

        h_target = np.zeros(n)
        for op in self.nucleus.ops_1_body:
            h_target[op.A] += op.g

        interactions = {}
        for op in self.nucleus.ops_hop:
            pair = (op.A, op.B)
            interactions.setdefault(pair, {"hopping": 0.0, "diag": 0.0})
            interactions[pair]["hopping"] += op.g
        for op in self.nucleus.ops_2_body:
            pair = (op.A, op.B)
            interactions.setdefault(pair, {"hopping": 0.0, "diag": 0.0})
            interactions[pair]["diag"] += op.g

        return h_driver, h_target, interactions

    # ------------------------------------------------------------------
    # Gate-angle conversions.
    #
    # qibo's RZ(theta) = exp(-i*theta/2*Z) and RXXYY(theta) = exp(-i*theta/4*(XX+YY))
    # already include the physicist's -i convention; CU1(theta) = exp(+i*theta*N_A*N_B)
    # does NOT (it is a phase gate), so it needs an explicit sign flip to implement
    # time evolution under +coef*N_A*N_B.
    # ------------------------------------------------------------------

    @staticmethod
    def _rz_theta(z_coef: float, dt: float) -> float:
        """Angle for RZ so that RZ(theta) = exp(-i*dt*z_coef*Z)."""
        return -2.0 * dt * (0.5 * z_coef)

    @staticmethod
    def _cu1_theta(diag_coef: float, dt: float) -> float:
        """Angle for CU1 so that CU1(theta) = exp(-i*dt*diag_coef*N_A*N_B)."""
        return -dt * diag_coef

    @staticmethod
    def _rxxyy_theta(hop_coef: float, dt: float) -> float:
        """Angle for RXXYY so that RXXYY(theta) = exp(-i*dt*0.5*hop_coef*(XX+YY))."""
        return 2.0 * dt * hop_coef

    # ------------------------------------------------------------------
    # Classical (matrix) reference evolution
    # ------------------------------------------------------------------

    def exact_evolution(self, steps: int, dt: float, schedule=None) -> dict:
        """
        Classically Trotter-evolves the reference state under the small-matrix
        H(s) = (1-s)*H_D + s*H_T (exact unitary per step, via scipy.linalg.expm),
        alongside H(s)'s instantaneous eigenvalues at every step - the reference
        calculation the circuit simulation in run()/build_circuit() is checked
        against.

        Args:
            steps (int): Number of annealing steps.
            dt (float): Time step per Trotter step.
            schedule (callable, optional): s(step, steps) -> float in [0, 1].
                Defaults to a linear schedule.

        Returns:
            dict with time_points, energies_expected, energies_instantaneous,
            final_state (in the small many-body basis) and fidelity to the exact
            ground state (nucleus.eig_vec[:, 0]).
        """
        schedule = schedule or _linear_schedule

        psi = np.zeros(self.H_D_matrix.shape[0], dtype=complex)
        psi[self.ref_state] = 1.0

        time_points = []
        energies_expected = []
        energies_instantaneous = []

        for step in range(steps + 1):
            s = schedule(step, steps)
            H_s = (1 - s) * self.H_D_matrix + s * self.H_T_matrix

            E_expect = (psi.conj().T @ (H_s @ psi)).real
            energies_expected.append(float(E_expect))
            energies_instantaneous.append(la.eigvalsh(H_s))
            time_points.append(step * dt)

            U = expm(-1j * H_s * dt)
            psi = U @ psi

        fidelity = abs(np.vdot(self.nucleus.eig_vec[:, 0], psi)) ** 2

        return {
            "time_points": np.array(time_points),
            "energies_expected": np.array(energies_expected),
            "energies_instantaneous": np.array(energies_instantaneous),
            "final_state": psi,
            "fidelity": float(fidelity),
        }

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------

    def build_circuit(self, steps: int, dt: float, schedule=None):
        """
        Builds the direct-gate Trotterized annealing circuit (arbitrary-qubit-pair
        connectivity - see build_swap_network_circuit for the nearest-neighbor
        HCB variant). Fermionic terms are applied via the staircase Pauli-
        exponential technique (Circuits_Composer.Qibo_staircase_pauli_exponential);
        HCB terms are applied via native RZ/CU1/RXXYY gates on their own qubit
        pair.
        """
        schedule = schedule or _linear_schedule
        circuit = self.Qibo_ref_state_composer()

        if self.encoding == "fermionic":
            for step in range(steps + 1):
                s = schedule(step, steps)
                H_s = (1 - s) * self.H_D_qubit + s * self.H_T_qubit
                self.Qibo_staircase_pauli_exponential(circuit, H_s, dt)
            return circuit

        h_driver, h_target, interactions = self._hcb_coefficients()
        for step in range(steps + 1):
            s = schedule(step, steps)
            for q in range(self.n_qubits):
                z_coef = (1 - s) * h_driver[q] + s * h_target[q]
                if abs(z_coef) > 1e-12:
                    circuit.add(gates.RZ(q, theta=self._rz_theta(z_coef, dt)))
            for (A, B), coeffs in interactions.items():
                hop = s * coeffs["hopping"]
                diag = s * coeffs["diag"]
                if abs(hop) > 1e-12:
                    circuit.add(gates.RXXYY(A, B, theta=self._rxxyy_theta(hop, dt)))
                if abs(diag) > 1e-12:
                    circuit.add(gates.CU1(A, B, theta=self._cu1_theta(diag, dt)))
        return circuit

    def build_swap_network_circuit(self, steps: int, dt: float, schedule=None):
        """
        Builds a nearest-neighbor "swap network" Trotterized annealing circuit for
        the HCB encoding: only ever applies 2-qubit gates between adjacent
        physical qubits, sweeping a logical-mode <-> physical-qubit map back and
        forth via SWAP gates so every pair of modes becomes adjacent at some point
        - the hardware-realistic connectivity model for linear-topology devices
        (translated from the reference notebook's apply_1body_layer/
        apply_interaction_gate/apply_swap_block_forward/apply_swap_block_reverse,
        sourced from nucleus.ops_1_body/ops_2_body/ops_hop instead of a separately
        re-derived Hamiltonian).
        """
        if self.encoding != "HCB":
            raise NotImplementedError(
                "The nearest-neighbor swap-network circuit is only implemented "
                "for the HCB encoding."
            )

        schedule = schedule or _linear_schedule
        h_driver, h_target, interactions = self._hcb_coefficients()
        n = self.n_qubits

        circuit = self.Qibo_ref_state_composer()
        # current_map[physical qubit] = logical mode currently held there.
        current_map = list(range(n))

        def apply_1body_layer(s):
            for phys_q in range(n):
                logic_q = current_map[phys_q]
                z_coef = (1 - s) * h_driver[logic_q] + s * h_target[logic_q]
                if abs(z_coef) > 1e-12:
                    circuit.add(gates.RZ(phys_q, theta=self._rz_theta(z_coef, dt)))

        def apply_interaction(q1, q2, s):
            l1, l2 = current_map[q1], current_map[q2]
            pair = (l1, l2) if l1 < l2 else (l2, l1)
            coeffs = interactions.get(pair)
            if coeffs is None:
                return
            hop = s * coeffs["hopping"]
            diag = s * coeffs["diag"]
            if abs(hop) > 1e-12:
                circuit.add(gates.RXXYY(q1, q2, theta=self._rxxyy_theta(hop, dt)))
            if abs(diag) > 1e-12:
                circuit.add(gates.CU1(q1, q2, theta=self._cu1_theta(diag, dt)))

        def swap_forward(q1, q2, s):
            apply_interaction(q1, q2, s)
            circuit.add(gates.SWAP(q1, q2))
            current_map[q1], current_map[q2] = current_map[q2], current_map[q1]

        def swap_reverse(q1, q2, s):
            circuit.add(gates.SWAP(q1, q2))
            current_map[q1], current_map[q2] = current_map[q2], current_map[q1]
            apply_interaction(q1, q2, s)

        swap_depth = n
        for step in range(steps):
            s = schedule(step + 1, steps)
            if step % 2 == 0:
                apply_1body_layer(s)
                for layer in range(swap_depth):
                    start_q = layer % 2
                    for q in range(start_q, n - 1, 2):
                        swap_forward(q, q + 1, s)
            else:
                for layer in reversed(range(swap_depth)):
                    start_q = layer % 2
                    for q in range(start_q, n - 1, 2):
                        swap_reverse(q, q + 1, s)
                apply_1body_layer(s)

        return circuit

    # ------------------------------------------------------------------
    # Full-Hilbert-space helpers
    # ------------------------------------------------------------------

    def embed_subspace_vector(self, vector) -> np.ndarray:
        """
        Embeds a vector expressed in the nucleus's small many-body basis
        (nucleus.states for fermionic, nucleus.hcb_states for HCB - e.g.
        nucleus.eig_vec[:, 0], the true ground state) into the full
        2**n_qubits computational-basis statevector, generalizing the reference
        notebook's per-nucleus hardcoded map_subspace_to_full_hilbert to any
        nucleus/encoding.
        """
        basis = (
            self.nucleus.states
            if self.encoding == "fermionic"
            else self.nucleus.hcb_states
        )
        if len(vector) != len(basis):
            raise ValueError(
                f"vector has length {len(vector)}, expected {len(basis)} (the "
                "many-body basis dimension)"
            )

        full = np.zeros(2**self.n_qubits, dtype=complex)
        for idx, occ in enumerate(basis):
            local_occ = (
                self._local_list(occ)
                if self.encoding == "fermionic"
                else list(occ)
            )
            bit_index = 0
            for q in local_occ:
                bit_index += 2 ** (self.n_qubits - 1 - q)
            full[bit_index] = vector[idx]
        return full

    # ------------------------------------------------------------------
    # Top-level run
    # ------------------------------------------------------------------

    def run(
        self, steps: int, dt: float, schedule=None, swap_network: bool = False
    ) -> dict:
        """
        Builds the requested circuit variant, executes it (exact statevector
        simulation), and returns its final state, energy and fidelity to the
        exact ground state.
        """
        if swap_network:
            circuit = self.build_swap_network_circuit(steps, dt, schedule=schedule)
        else:
            circuit = self.build_circuit(steps, dt, schedule=schedule)

        result = circuit()
        state = np.array(result.state())

        H_T_full = get_sparse_operator(self.H_T_qubit, n_qubits=self.n_qubits)
        energy = (state.conj().T @ (H_T_full @ state)).real

        ground_state_full = self.embed_subspace_vector(self.nucleus.eig_vec[:, 0])
        fidelity = abs(np.vdot(ground_state_full, state)) ** 2

        return {
            "state": state,
            "energy": float(energy),
            "fidelity": float(fidelity),
            "circuit": circuit,
        }


def Annealing_simulation(
    nucleus: str,
    ref_state: int = 0,
    shell: str = "p",
    steps: int = 100,
    dt: float = 0.1,
    encoding: str = "fermionic",
    swap_network: bool = False,
    schedule=None,
):
    """
    Convenience entry point mirroring Methods.py's *_minimization() pattern:
    builds the Nucleus, wraps it in an AnnealingProtocol, runs the annealing
    circuit, and returns (result, nucleus).
    """
    nuc = Nucleus(nucleus, shell=shell, encoding=encoding)
    protocol = AnnealingProtocol(nuc, ref_state=ref_state, encoding=encoding)
    result = protocol.run(steps, dt, schedule=schedule, swap_network=swap_network)
    result["protocol"] = protocol
    return result, nuc

import numpy as np
from numpy import linalg as la
import scipy.sparse as sp
import os


class TwoBodyExcitationOperator:
    """
    Class to define an antihermitian operator corresponding to a two-body excitation. The data of the operator
    is taken from already processed files in the data folder for each nucleus.

    Attributes:

    """

    def __init__(
        self,
        label: int,
        H2b: float,
        ijkl: list,
        matrix: np.ndarray,
        commutator: np.ndarray,
    ) -> None:
        """
        Initializes the operator instance, according to its pre-generated data.

        Args:
            label (int): Label of the operator (as it appears on the data files).
            H2b (float): Value of the amplitude of the operator in the hamiltonian.
            ijkl (list): Indices of the two body excitation: (a'_i a'_j a_k a_l).
            matrix (np.ndarray): Matrix representation of the operator.
            commutator (np.ndarray): Commutator of the operator with the Hamiltonian ([H, T]).
        """
        self.label = label
        self.H2b = H2b
        self.ijkl = ijkl
        self.matrix = matrix
        self.commutator = commutator


class OneBodyExcitationOperator:
    """
    Class to define an antihermitian operator corresponding to one-body excitation. The data of the operator
    is taken from already processed files in the data folder for each nucleus.

    Attributes:

    """

    def __init__(self, eps: float, i: int, matrix: np.ndarray) -> None:
        """
        Initializes the operator instance, according to its pre-generated data.

        Args:
            eps (float): Value of the amplitude of the operator in the hamiltonian.
            i (list): Indices of the two body excitation: (a'_i a_i).
            matrix (np.ndarray): Matrix representation of the operator.

        """

        self.eps = eps
        self.i = i
        self.matrix = matrix


class HCB_OneBodyOperator:
    """One-body diagonal operator for the HCB (hard-core boson) encoding."""

    def __init__(self, A: int, g: float, matrix: np.ndarray) -> None:
        self.A = A
        self.g = g
        self.matrix = matrix


class HCB_TwoBodyDiagonalOperator:
    """Two-body diagonal operator for the HCB (hard-core boson) encoding."""

    def __init__(self, A: int, B: int, g: float, matrix: np.ndarray) -> None:
        self.A = A
        self.B = B
        self.g = g
        self.matrix = matrix


class HCB_HoppingOperator:
    """Two-body hopping (hard-core boson pair-transfer) operator for the HCB encoding."""

    def __init__(
        self, A: int, B: int, g: float, matrix: np.ndarray, op_matrix, commutator
    ) -> None:
        self.A = A
        self.B = B
        self.g = g
        self.matrix = matrix
        self.op_matrix = op_matrix
        self.commutator = commutator


class Nucleus:
    """
    Class to define a nucleus: its Hamiltonian, many-body basis and excitation
    operator pool, for either the fermionic or the HCB (hard-core boson) encoding.

    Args:
        nuc_name (str): Name of the nucleus.
        shell (str): Valence shell ('p' or 'sd').
        encoding (str): Encoding used to represent the system, either 'fermionic' or 'HCB'.
            For 'HCB', the fermionic single-particle basis is built first (as `self.nucleus`)
            and the hard-core boson mapping and operators are derived analytically from it.
    """

    def __init__(
        self, nuc_name: str, shell: str = "p", encoding: str = "fermionic"
    ) -> None:
        self.encoding = encoding

        if encoding == "fermionic":
            self.name = nuc_name
            self.shell = shell.lower()
            self.data_folder = os.path.join(f"nuclei/{self.name}_data")

            # Load the basis first, needed for the space analysis below
            self.states = self.states_list()
            # O(1) state -> index lookup, instead of repeated O(d_H) list.index()/`in` scans
            self._state_index = {state: idx for idx, state in enumerate(self.states)}

            # Shell structure definition
            self.shell_config = {
                "p": {"p_range": range(0, 6), "n_range": range(6, 12), "size": 6},
                "sd": {"p_range": range(0, 12), "n_range": range(12, 24), "size": 12},
            }

            range_qubits, self.active_indices = self._analyze_space()
            self.qubits = range(range_qubits[0], range_qubits[1] + 1)

            self.H = self.hamiltonian_matrix()
            self.d_H = self.H.shape[0]
            self.eig_val, self.eig_vec = la.eigh(self.H)
            self.operators = self.operators_list()

            # Memoization caches for Ham_1_body_contributions()/Ham_2_body_contributions():
            # these are only needed for post-hoc reporting (not during the VQE loop
            # itself), but are frequently called more than once on the same Nucleus
            # (e.g. once by Circuits_Composer, once by the *_minimization() callers),
            # each time redoing an O(n_ops * d_H) dense matrix-construction loop.
            self._ham1_cache = None
            self._ham2_cache = None

        elif encoding == "HCB":
            self.nucleus = Nucleus(nuc_name, shell=shell, encoding="fermionic")
            self.name = nuc_name
            self.shell = shell

            self.hcb_mapping = self._build_hcb_mapping()
            self.hcb_states = self._build_hcb_states()

            # Pre-load raw Hamiltonian parameters for analytical calculation
            self.sp_energies = self._load_sp_energies()
            self.v_2b = self._load_two_body_interaction()

            self.H = self.build_hcb_hamiltonian()
            self.d_H = self.H.shape[0]
            self.eig_val, self.eig_vec = la.eigh(self.H)
            self.ops_1_body, self.ops_2_body, self.ops_hop = self.build_hcb_operators()

    # ------------------------------------------------------------------
    # Fermionic encoding
    # ------------------------------------------------------------------

    def _analyze_space(self) -> tuple:
        """
        Determines the range of active qubits as a (min, max) tuple and
        filters out the inactive orbitals.
        """
        conf = self.shell_config.get(self.shell, self.shell_config["p"])

        all_indices = [idx for state in self.states for idx in state]
        if not all_indices:
            return (0, 0), []

        has_p = any(i in conf["p_range"] for i in all_indices)
        has_n = any(i in conf["n_range"] for i in all_indices)

        # Limits tuple, defined according to occupation and shell
        p_min, p_max = conf["p_range"].start, conf["p_range"].stop - 1
        n_min, n_max = conf["n_range"].start, conf["n_range"].stop - 1

        if has_p and not has_n:
            qubit_range = (p_min, p_max)
        elif has_n and not has_p:
            qubit_range = (n_min, n_max)
        else:
            qubit_range = (p_min, n_max)

        active_indices = []
        total_states = len(self.states)

        # Only consider orbitals within the identified range
        for i in range(qubit_range[0], qubit_range[1] + 1):
            occ_count = sum(1 for state in self.states if i in state)
            if 0 < occ_count < total_states:
                active_indices.append(i)

        qubit_range = (active_indices[0], active_indices[-1])

        return qubit_range, active_indices

    def Ham_1_body_contributions(self) -> list:
        """
        Returns the list of ALL hermitian observables corresponding to one-body excitations.
        The indices of the avaliable operators are taken from the data files of the nucleus, since it only includes
        those operators that respect the selection rules.

        Returns:
            list[OneBodyExcitationOperators]: List of hermitian operators, as OneBodyExcitationOperator instances.
        """

        if self._ham1_cache is not None:
            return self._ham1_cache

        operators = []
        sp_path = os.path.join(self.data_folder, f"sp.dat")
        sp_data = np.loadtxt(sp_path, dtype=str, skiprows=1)

        states = self.states
        monoparticular_energies = {}

        for row in sp_data:
            sp = row[0]
            if int(sp) in self.qubits:
                eps = float(row[-1])
                monoparticular_energies[sp] = eps

        for sp_index, eps in monoparticular_energies.items():
            if int(sp_index) in self.qubits:
                matrix = np.zeros((self.d_H, self.d_H))

                for state_index, state in enumerate(states):

                    if int(sp_index) in state:
                        matrix[state_index, state_index] = 1

                operator = OneBodyExcitationOperator(
                    eps=eps, i=int(sp_index), matrix=matrix
                )
                operators.append(operator)

        self._ham1_cache = (operators, monoparticular_energies)
        return self._ham1_cache

    def states_list(self) -> list:
        """
        Returns the list of states of the many-body basis according to the indices of the single-particle states.

        Returns:
            list: List of the basis states.
        """
        states = []
        mb_path = os.path.join(self.data_folder, f"mb_basis_2.dat")
        file = open(mb_path, "r")
        self.d_H = int(file.readline().strip())
        mb_data = np.loadtxt(mb_path, dtype=str, delimiter=" ", skiprows=1)
        for m in mb_data:
            sp_labels = []
            for i in range(1, len(m)):
                label = int(m[i].replace(",", "").replace("(", "").replace(")", ""))
                sp_labels.append(int(label))
            states.append(tuple(sp_labels))
        return states

    def Ham_2_body_contributions(self) -> list:
        """
        Returns the list of ALL hermitian observables corresponding to two-body excitations.
        The indices of the avaliable operators are taken from the data files of the nucleus, since it only includes
        those operators that respect the selection rules.

        Returns:
            list[TwoBodyExcitationOperators]: List of antihermitian operators, as TwoBodyExcitationOperator instances.
        """
        if self._ham2_cache is not None:
            return self._ham2_cache

        operators = []
        H2b_path = os.path.join(self.data_folder, f"H2b.dat")
        H2b_data = np.loadtxt(H2b_path, dtype=str)
        label = 1
        for h in H2b_data:
            indices = [int(h[1]), int(h[2]), int(h[3]), int(h[4])]
            i, j, k, l = indices
            H2b = float(h[0])
            if max(indices) in self.qubits and min(indices) in self.qubits:
                if i < j and k < l and (i, j) <= (k, l):
                    operator_matrix = np.zeros((self.d_H, self.d_H))
                    for column, state in enumerate(self.states):
                        new_state, parity = self.excitation_numbers(state, indices)
                        row = self._state_index.get(new_state)
                        if row is not None:
                            operator_matrix[row, column] += parity
                            if row != column:
                                operator_matrix[column, row] += parity
                    commutator = 0

                    operators.append(
                        TwoBodyExcitationOperator(
                            label, H2b, indices, operator_matrix, commutator
                        )
                    )
                    label += 1
        self._ham2_cache = operators
        return operators

    def excitation_numbers(self, state: tuple, indices: list) -> tuple:
        """
        Returns the new state and parity of the excitation after the action of a two-body excitation operator
        on a state of the basis.

        Args:
            state (tuple): Indices of the single-particle states of a given state of the many-body basis.
            indices (list): Indices of the two-body excitation: (a_i* a_j* a_k a_l).
        Returns:
            tuple: New state after the excitation.
            int: Parity of the excitation.
        """
        parity = 1
        if indices[2] in state and indices[3] in state:
            new_state = list(state)
            # Annihilation
            for i in [indices[2], indices[3]]:
                parity *= (-1) ** (new_state.index(i))
                new_state.remove(i)
            # Creation
            for i in [indices[1], indices[0]]:
                new_state.append(i)
                new_state.sort()
                parity *= (-1) ** (new_state.index(i))

            return tuple(new_state), parity
        else:
            return tuple(), 0

    def operators_list(self) -> list:
        """
        Returns the list of ALL antihermitian operators corresponding to two-body excitations.
        The indices of the avaliable operators are taken from the data files of the nucleus, since it only includes
        those operators that respect the selection rules.

        Returns:
            list[TwoBodyExcitationOperators]: List of antihermitian operators, as TwoBodyExcitationOperator instances.
        """
        operators = []
        H2b_path = os.path.join(self.data_folder, f"H2b.dat")
        H2b_data = np.loadtxt(H2b_path, dtype=str)
        label = 1
        for h in H2b_data:
            indices = [int(h[1]), int(h[2]), int(h[3]), int(h[4])]
            i, j, k, l = indices
            if all(x in self.qubits for x in (i, j, k, l)):
                H2b = float(h[0])
                if i < j and k < l and (i, j) <= (k, l):
                    operator_matrix = np.zeros((self.d_H, self.d_H))
                    for column, state in enumerate(self.states):
                        new_state, parity = self.excitation_numbers(state, [i, j, l, k])
                        row = self._state_index.get(new_state)
                        if row is not None:
                            operator_matrix[row, column] += parity
                            operator_matrix[column, row] += -parity

                    if operator_matrix.any():
                        # operator_matrix is extremely sparse (a handful of nonzero
                        # entries even when d_H is in the hundreds), so building the
                        # commutator through a sparse intermediate avoids the dense
                        # O(d_H^3) matrix product; the result is still returned dense.
                        op_sparse = sp.csr_matrix(operator_matrix)
                        commutator = self.H @ op_sparse - op_sparse @ self.H
                        operators.append(
                            TwoBodyExcitationOperator(
                                label, H2b, indices, operator_matrix, commutator
                            )
                        )
                        label += 1
        return operators

    def hamiltonian_matrix(self) -> np.ndarray:
        """
        Returns the hamiltonian matrix of the nucleus.

        Returns:
            np.ndarray: Hamiltonian matrix.
        """
        file_path = os.path.join(self.data_folder, f"{self.name}.dat")
        H = np.zeros((self.d_H, self.d_H))
        H_data = np.loadtxt(file_path, delimiter=" ", dtype=float)
        for line in H_data:
            H[int(line[0]), int(line[1])] = line[2]
        return H

    # ------------------------------------------------------------------
    # HCB (hard-core boson) encoding
    # ------------------------------------------------------------------

    def _build_hcb_mapping(self) -> dict:
        """Build mapping from single-particle states to hard-core boson modes."""
        sp_path = os.path.join(self.nucleus.data_folder, "sp.dat")
        sp_data = np.loadtxt(sp_path, dtype=str, skiprows=1)

        hcb_mapping = {}
        hcb_index = 0

        states_by_nlj = {}
        for row in sp_data:
            i = int(row[0])

            # --- CRITICAL FIX ---
            # Skip states outside the active qubit range, so we only ever
            # build pairs within the active space.
            if i not in self.nucleus.qubits:
                continue
            # ---------------------------

            n = int(row[1])
            l = int(row[2])
            j = float(row[3])
            m = float(row[4])
            tz = row[5]

            key = (n, l, j, tz)
            if key not in states_by_nlj:
                states_by_nlj[key] = {}
            states_by_nlj[key][m] = i

        # Create hard-core boson pairs
        for (n, l, j, tz), m_dict in states_by_nlj.items():
            m_values = sorted([m for m in m_dict.keys() if m > 0], reverse=True)
            for m in m_values:
                if m in m_dict and -m in m_dict:
                    hcb_mapping[hcb_index] = {
                        "i_plus": m_dict[m],  # a
                        "i_minus": m_dict[-m],  # ã
                        "j": j,
                        "tz": tz,
                    }
                    hcb_index += 1

        return hcb_mapping

    def _build_hcb_states(self) -> list:
        """Build hard-core boson many-body states from fermionic states."""
        hcb_states = []
        for state in self.nucleus.states:
            hcb_occupation = []
            state_list = list(state)
            valid_hcb_state = True

            for A, hcb_info in self.hcb_mapping.items():
                i_plus = hcb_info["i_plus"]
                i_minus = hcb_info["i_minus"]

                has_plus = i_plus in state_list
                has_minus = i_minus in state_list

                if has_plus and has_minus:
                    hcb_occupation.append(A)
                    state_list.remove(i_plus)
                    state_list.remove(i_minus)
                elif has_plus or has_minus:
                    valid_hcb_state = False
                    break

            if valid_hcb_state and len(state_list) == 0:
                hcb_states.append(tuple(hcb_occupation))
        return hcb_states

    def _load_sp_energies(self) -> dict:
        """Loads single particle energies epsilon_a into a dictionary."""
        sp_path = os.path.join(self.nucleus.data_folder, "sp.dat")
        sp_data = np.loadtxt(sp_path, dtype=str, skiprows=1)
        energies = {}
        for row in sp_data:
            idx = int(row[0])
            # Check if index is within qubit range
            if idx in self.nucleus.qubits:
                energies[idx] = float(row[-1])
        return energies

    def _load_two_body_interaction(self) -> dict:
        """Loads two-body matrix elements V_ijkl into a dictionary for O(1) access."""
        H2b_path = os.path.join(self.nucleus.data_folder, "H2b.dat")
        H2b_data = np.loadtxt(H2b_path, dtype=str)
        v_2b = {}
        for h in H2b_data:
            val = float(h[0])
            i, j, k, l = int(h[1]), int(h[2]), int(h[3]), int(h[4])
            # Store exactly as in file (usually canonical ordering)
            v_2b[(i, j, k, l)] = val
        return v_2b

    def _get_v(self, i: int, j: int, k: int, l: int) -> float:
        """
        Robustly retrieves V_ijkl handling index symmetries.
        Assumes real, anti-symmetric matrix elements:
        V_ijkl = -V_jikl = -V_ijlk = V_jilk
        V_ijkl = V_klij (hermiticity/real)
        """
        sign = 1.0

        # Canonicalize pairs (i, j) and (k, l) to be ascending
        if i > j:
            i, j = j, i
            sign *= -1.0
        if k > l:
            k, l = l, k
            sign *= -1.0

        # Canonicalize the order of pairs
        # We need (i,j) <= (k,l) to match standard storage formats
        if (i, j) > (k, l):
            i, j, k, l = k, l, i, j
            # No sign change for swapping pairs in real hermitian interaction

        return sign * self.v_2b.get((i, j, k, l), 0.0)

    def build_hcb_operators(self) -> tuple:
        """
        Builds the Hamiltonian operators using analytical formulas.
        Constructs H_Q matrix merely for consistency/verification if needed.
        """
        d_hcb = len(self.hcb_states)

        # Clear lists
        self.ops_1_body = []
        self.ops_2_body = []
        self.ops_hop = []

        n_hcb = len(self.hcb_mapping)

        # 1. ONE-BODY TERMS: g_AA
        for A in range(n_hcb):
            # Retrieve indices for a and a_tilde
            a = self.hcb_mapping[A]["i_plus"]
            a_tilde = self.hcb_mapping[A]["i_minus"]

            eps_a = self.sp_energies.get(a, 0.0)

            # Formula: g_AA = 4*eps_a - V_{a a~ a a~}
            # Note: Using get_v ensures we find V(a, a~, a, a~) regardless of stored order
            v_term = self._get_v(a, a_tilde, a, a_tilde)

            g_AA = 2 * eps_a + v_term

            # Build diagonal matrix for this operator
            matrix = np.zeros((d_hcb, d_hcb))
            for idx, state in enumerate(self.hcb_states):
                if A in state:
                    matrix[idx, idx] = 1.0

            if abs(g_AA) > 1e-10:
                self.ops_1_body.append(HCB_OneBodyOperator(A, g_AA, matrix))

        # 2. TWO-BODY TERMS
        for A in range(n_hcb):
            for B in range(A + 1, n_hcb):
                # Indices
                a = self.hcb_mapping[A]["i_plus"]
                a_tilde = self.hcb_mapping[A]["i_minus"]
                b = self.hcb_mapping[B]["i_plus"]
                b_tilde = self.hcb_mapping[B]["i_minus"]

                # --- Diagonal Term: g_ABAB ---
                # Formula: g_ABAB = V_{b a b a} + V_{a~ b a~ b} + V_{a b~ a b~} + V_{a~ b~ a~ b~}
                # Note: We simply sum the four exchange-like terms
                term1 = self._get_v(b, a, b, a)
                term2 = self._get_v(a_tilde, b, a_tilde, b)
                term3 = self._get_v(a, b_tilde, a, b_tilde)
                term4 = self._get_v(a_tilde, b_tilde, a_tilde, b_tilde)

                g_diag = term1 + term2 + term3 + term4

                diag_matrix = np.zeros((d_hcb, d_hcb))
                for idx, state in enumerate(self.hcb_states):
                    if A in state and B in state:
                        diag_matrix[idx, idx] = 1.0

                if abs(g_diag) > 1e-10 and np.any(diag_matrix):
                    self.ops_2_body.append(
                        HCB_TwoBodyDiagonalOperator(A, B, g_diag, diag_matrix)
                    )

                g_hop = self._get_v(a, a_tilde, b, b_tilde)

                hop_matrix = self._build_hopping_matrix(A, B)
                hop_op_matrix = self._build_hopping_op_matrix(A, B)
                # hop_op_matrix is extremely sparse (a handful of nonzero entries even
                # when d_hcb is large), so building the commutator through a sparse
                # intermediate avoids the dense O(d_hcb^3) matrix product.
                hop_op_sparse = sp.csr_matrix(hop_op_matrix)
                commutator = self.H @ hop_op_sparse - hop_op_sparse @ self.H

                if abs(g_hop) > 1e-10 and np.any(hop_matrix):
                    self.ops_hop.append(
                        HCB_HoppingOperator(
                            A, B, g_hop, hop_matrix, hop_op_matrix, commutator
                        )
                    )

        return self.ops_1_body, self.ops_2_body, self.ops_hop

    def _build_hopping_matrix(self, A: int, B: int) -> np.ndarray:
        """Build matrix for hopping operator S_A^+ S_B^- + S_A^- S_B^+."""
        d_hcb = len(self.hcb_states)
        matrix = np.zeros((d_hcb, d_hcb))

        for i, state_i in enumerate(self.hcb_states):
            for j, state_j in enumerate(self.hcb_states):
                # S_A^+ S_B^-: create A, destroy B (Moves pair from B to A)
                if A not in state_i and B in state_i:
                    state_new = tuple(sorted(set(state_i) - {B} | {A}))
                    if state_new == state_j:
                        matrix[j, i] += 1.0

                # S_A^- S_B^+: destroy A, create B (Moves pair from A to B)
                if A in state_i and B not in state_i:
                    state_new = tuple(sorted(set(state_i) - {A} | {B}))
                    if state_new == state_j:
                        matrix[j, i] += 1.0
        return matrix

    def _build_hopping_op_matrix(self, A: int, B: int) -> np.ndarray:
        """Build matrix for hopping operator S_A^+ S_B^- + S_A^- S_B^+."""
        d_hcb = len(self.hcb_states)
        matrix = np.zeros((d_hcb, d_hcb))

        for i, state_i in enumerate(self.hcb_states):
            for j, state_j in enumerate(self.hcb_states):
                # S_A^+ S_B^-: create A, destroy B (Moves pair from B to A)
                if A not in state_i and B in state_i:
                    state_new = tuple(sorted(set(state_i) - {B} | {A}))
                    if state_new == state_j:
                        matrix[j, i] += 1.0

                # S_A^- S_B^+: destroy A, create B (Moves pair from A to B)
                if A in state_i and B not in state_i:
                    state_new = tuple(sorted(set(state_i) - {A} | {B}))
                    if state_new == state_j:
                        matrix[j, i] -= 1.0
        return matrix

    def _hcb_state_to_fermionic(self, hcb_state: tuple) -> tuple:
        """Convert a hard-core boson state to its fermionic state."""
        fermions = []
        for A in hcb_state:
            hcb_info = self.hcb_mapping[A]
            fermions.extend([hcb_info["i_plus"], hcb_info["i_minus"]])
        return tuple(sorted(fermions))

    def build_hcb_hamiltonian(self) -> tuple:
        """
        Build the HCB Hamiltonian H_Q by projecting H_NSM.

        Returns:
            tuple: (H_Q matrix, one-body ops, two-body diagonal ops, hopping ops)
        """
        d_hcb = len(self.hcb_states)
        H_Q = np.zeros((d_hcb, d_hcb))

        # Precompute each HCB state's fermionic-basis index once (O(1) dict lookup)
        # instead of recomputing/re-searching it on every (i, j) pair below.
        ferm_indices = [
            self.nucleus._state_index.get(self._hcb_state_to_fermionic(state))
            for state in self.hcb_states
        ]

        # Project H_NSM onto the HCB subspace
        for i, idx_i in enumerate(ferm_indices):
            if idx_i is None:
                continue
            for j, idx_j in enumerate(ferm_indices):
                if idx_j is None:
                    continue
                H_Q[i, j] = self.nucleus.H[idx_i, idx_j]

        return H_Q

import numpy as np
from numpy import linalg as la
import os

from ADAPT_VQE.Nucleus import Nucleus

class QuasiparticleOneBodyOperator():
    def __init__(self, A: int, g: float, matrix: np.ndarray) -> None:
        self.A = A
        self.g = g
        self.matrix = matrix

class QuasiparticleTwoBodyDiagonal():
    def __init__(self, A: int, B: int, g: float, matrix: np.ndarray) -> None:
        self.A = A
        self.B = B
        self.g = g
        self.matrix = matrix

class QuasiparticleHopping():
    def __init__(self, A: int, B: int, g: float, matrix: np.ndarray, op_matrix, commutator) -> None:
        self.A = A
        self.B = B
        self.g = g
        self.matrix = matrix
        self.op_matrix = op_matrix
        self.commutator = commutator

class QuasiparticleNucleus():
    """
    Extension of Nucleus class to handle quasiparticle encoding using analytical
    derivation of coefficients from the raw Hamiltonian parameters.
    """
    
    def __init__(self, nucleus: str, n_qubits: int = 6) -> None:
        self.nucleus = Nucleus(nucleus, n_qubits=n_qubits)
        self.name = nucleus
        self.n_qubits = n_qubits
        self.qp_mapping = self._build_qp_mapping()
        self.qp_states = self._build_qp_states()
        
        # Pre-load raw Hamiltonian parameters for analytical calculation
        self.sp_energies = self._load_sp_energies()
        self.v_2b = self._load_two_body_interaction()
        
        
        self.H = self.build_qp_hamiltonian()
        self.d_H = self.H.shape[0] 
        self.eig_val, self.eig_vec = la.eigh(self.H)
        self.ops_1_body, self.ops_2_body, self.ops_hop = self.build_qp_operators()
        
    def _build_qp_mapping(self) -> dict:
        """Build mapping from single-particle states to quasiparticle modes."""
        sp_path = os.path.join(self.nucleus.data_folder, 'sp.dat')
        sp_data = np.loadtxt(sp_path, dtype=str, skiprows=1)
        
        qp_mapping = {}
        qp_index = 0
        
        states_by_nlj = {}
        for row in sp_data:
            i = int(row[0])
            
            # --- CORRECCIÓN CRÍTICA ---
            # Si el índice del estado es mayor o igual a n_qubits, lo ignoramos.
            # Esto asegura que solo creemos pares dentro del espacio activo.
            if i >= self.nucleus.n_qubits:
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
        
        # Create quasiparticle pairs
        for (n, l, j, tz), m_dict in states_by_nlj.items():
            m_values = sorted([m for m in m_dict.keys() if m > 0], reverse=True)
            for m in m_values:
                if m in m_dict and -m in m_dict:
                    qp_mapping[qp_index] = {
                        'i_plus': m_dict[m],     # a
                        'i_minus': m_dict[-m],   # ã
                        'j': j,
                        'tz': tz
                    }
                    qp_index += 1
        
        return qp_mapping
    
    def _build_qp_states(self) -> list:
        """Build quasiparticle many-body states from fermionic states."""
        qp_states = []
        for state in self.nucleus.states:
            qp_occupation = []
            state_list = list(state)
            valid_qp_state = True
            
            for A, qp_info in self.qp_mapping.items():
                i_plus = qp_info['i_plus']
                i_minus = qp_info['i_minus']
                
                has_plus = i_plus in state_list
                has_minus = i_minus in state_list
                
                if has_plus and has_minus:
                    qp_occupation.append(A)
                    state_list.remove(i_plus)
                    state_list.remove(i_minus)
                elif has_plus or has_minus:
                    valid_qp_state = False
                    break
            
            if valid_qp_state and len(state_list) == 0:
                qp_states.append(tuple(qp_occupation))
        return qp_states

    def _load_sp_energies(self) -> dict:
        """Loads single particle energies epsilon_a into a dictionary."""
        sp_path = os.path.join(self.nucleus.data_folder, 'sp.dat')
        sp_data = np.loadtxt(sp_path, dtype=str, skiprows=1)
        energies = {}
        for row in sp_data:
            idx = int(row[0])
            # Check if index is within qubit range
            if idx < self.nucleus.n_qubits:
                energies[idx] = float(row[-1])
        return energies

    def _load_two_body_interaction(self) -> dict:
        """Loads two-body matrix elements V_ijkl into a dictionary for O(1) access."""
        H2b_path = os.path.join(self.nucleus.data_folder, 'H2b.dat')
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

    def build_qp_operators(self) -> tuple:
        """
        Builds the Hamiltonian operators using analytical formulas.
        Constructs H_Q matrix merely for consistency/verification if needed.
        """
        d_qp = len(self.qp_states)
        
        # Clear lists
        self.ops_1_body = []
        self.ops_2_body = []
        self.ops_hop = []
        
        n_qp = len(self.qp_mapping)
        
        # 1. ONE-BODY TERMS: g_AA
        for A in range(n_qp):
            # Retrieve indices for a and a_tilde
            a = self.qp_mapping[A]['i_plus']
            a_tilde = self.qp_mapping[A]['i_minus']
            
            eps_a = self.sp_energies.get(a, 0.0)
            
            # Formula: g_AA = 4*eps_a - V_{a a~ a a~}
            # Note: Using get_v ensures we find V(a, a~, a, a~) regardless of stored order
            v_term = self._get_v(a, a_tilde, a, a_tilde)
            
            g_AA = 2 * eps_a + v_term
            
            # Build diagonal matrix for this operator
            matrix = np.zeros((d_qp, d_qp))
            for idx, state in enumerate(self.qp_states):
                if A in state:
                    matrix[idx, idx] = 1.0
                    
            if abs(g_AA) > 1e-10:
                self.ops_1_body.append(
                    QuasiparticleOneBodyOperator(A, g_AA, matrix)
                )
            

        # 2. TWO-BODY TERMS
        for A in range(n_qp):
            for B in range(A + 1, n_qp):
                # Indices
                a = self.qp_mapping[A]['i_plus']
                a_tilde = self.qp_mapping[A]['i_minus']
                b = self.qp_mapping[B]['i_plus']
                b_tilde = self.qp_mapping[B]['i_minus']

                # --- Diagonal Term: g_ABAB ---
                # Formula: g_ABAB = V_{b a b a} + V_{a~ b a~ b} + V_{a b~ a b~} + V_{a~ b~ a~ b~}
                # Note: We simply sum the four exchange-like terms
                term1 = self._get_v(b, a, b, a)
                term2 = self._get_v(a_tilde, b, a_tilde, b)
                term3 = self._get_v(a, b_tilde, a, b_tilde)
                term4 = self._get_v(a_tilde, b_tilde, a_tilde, b_tilde)
                
                g_diag = term1 + term2 + term3 + term4
                
                diag_matrix = np.zeros((d_qp, d_qp))
                for idx, state in enumerate(self.qp_states):
                    if A in state and B in state:
                        diag_matrix[idx, idx] = 1.0
                
                if abs(g_diag) > 1e-10 and np.any(diag_matrix):
                    self.ops_2_body.append(
                        QuasiparticleTwoBodyDiagonal(A, B, g_diag, diag_matrix)
                    )
                
                
                g_hop =  self._get_v(a, a_tilde, b, b_tilde)
                
                hop_matrix = self._build_hopping_matrix(A, B)
                hop_op_matrix = self._build_hopping_op_matrix(A, B)
                commutator = self.H @ hop_op_matrix - hop_op_matrix @ self.H
                
                if abs(g_hop) > 1e-10 and np.any(hop_matrix):
                    self.ops_hop.append(
                        QuasiparticleHopping(A, B, g_hop, hop_matrix, hop_op_matrix, commutator)
                    )
                
            

        return self.ops_1_body, self.ops_2_body, self.ops_hop

    def _build_hopping_matrix(self, A: int, B: int) -> np.ndarray:
        """Build matrix for hopping operator S_A^+ S_B^- + S_A^- S_B^+."""
        d_qp = len(self.qp_states)
        matrix = np.zeros((d_qp, d_qp))
        
        for i, state_i in enumerate(self.qp_states):
            for j, state_j in enumerate(self.qp_states):
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
        d_qp = len(self.qp_states)
        matrix = np.zeros((d_qp, d_qp))
        
        for i, state_i in enumerate(self.qp_states):
            for j, state_j in enumerate(self.qp_states):
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
    
    def _qp_to_fermionic_state(self, qp_state: tuple) -> tuple:
        """Convert quasiparticle state to fermionic state."""
        fermions = []
        for A in qp_state:
            qp_info = self.qp_mapping[A]
            fermions.extend([qp_info['i_plus'], qp_info['i_minus']])
        return tuple(sorted(fermions))
    
    def build_qp_hamiltonian(self) -> tuple:
        """
        Build the quasiparticle Hamiltonian H_Q by projecting H_NSM.
        
        Returns:
            tuple: (H_Q matrix, one-body ops, two-body diagonal ops, hopping ops)
        """
        d_qp = len(self.qp_states)
        H_Q = np.zeros((d_qp, d_qp))
        
        # Project H_NSM onto quasiparticle subspace
        for i, qp_state_i in enumerate(self.qp_states):
            # Find corresponding fermionic state
            ferm_state_i = self._qp_to_fermionic_state(qp_state_i)
            if ferm_state_i not in self.nucleus.states:
                continue
            idx_i = self.nucleus.states.index(ferm_state_i)
            
            for j, qp_state_j in enumerate(self.qp_states):
                ferm_state_j = self._qp_to_fermionic_state(qp_state_j)
                if ferm_state_j not in self.nucleus.states:
                    continue
                idx_j = self.nucleus.states.index(ferm_state_j)
                
                H_Q[i, j] = self.nucleus.H[idx_i, idx_j]
        
        # Extract operator components
        
        
        return H_Q
    
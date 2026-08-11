from  scipy.sparse.linalg import expm_multiply
from QP_ADAPT_VQE.Nucleus import QuasiparticleNucleus
from QP_ADAPT_VQE.Circuit import QP_Circuits_Composser

import contextlib
import io
import numpy as np
from scipy.optimize import minimize

class Ansatz():
    """
    Parent class to define ansätze for VQE.

    Attributes:
        nucleus (Nucleus): Nucleus object.
        ref_state (np.ndarray): Reference state of the ansatz.
        all_operators (list): List of all the avaliable two-body excitation operators for a given nucleus.
        operator_pool (list): List of operators used in the ansatz.
        fcalls (int): Number of function calls of a VQE procedure.
        count_fcalls (bool): If True, the number of function calls during a VQE procedure is counted.
        ansatz (np.ndarray): Ansatz state.
    
    Methods:
        reduce_operators: Returns the list of operators excluding the repeated excitations.
        only_acting_operators: Returns the list of operators, only including the ones that have a non-zero action on the ref. state.
    """
    def __init__(self,
                 nucleus: QuasiparticleNucleus,
                 ref_state: np.ndarray) -> None:
        """
        Initialization of the Ansatz object.

        Args:
            nucleus (Nucleus): Nucleus object.
            ref_state (np.ndarray): Reference state of the ansatz.
            pool_format (str): Format of the operator pool. Available formats ['All', 'Reduced', 'Only acting', 'Custom'].
            operators_list (list): List of operators to be used in the ansatz, in case the pool format is 'Custom'.
        """
        self.nucleus = nucleus
        self.ref_state = ref_state
        
        self.operator_pool = nucleus.ops_hop
        
        self.fcalls = 0
        self.count_fcalls = False
        self.ansatz = self.ref_state
  
class QP_ADAPTAnsatz(Ansatz):
    """
    Child Ansatz class to define the ADAPT ansatz for VQE.

    Attributes:
        nucleus (Nucleus): Nucleus object.
        ref_state (np.ndarray): Reference state of the ansatz.
        pool_format (str): Format of the operator pool.
        operators_list (list): List of operators to be used in the ansatz.
        added_operators (list): List of operators added to the ansatz.
        minimum (bool): If True, the ansatz has reached the minimum energy.
        E0 (float): Energy of the ansatz without any excitation operators.

    Methods:
        build_ansatz: Returns the state of the ansatz on a given VQE iteratioin, after building it with the given paramters and the operators in the pool.
        energy: Returns the energy of the ansatz on a given VQE iteration.
        choose_operator: Returns the next operator and its gradient, after an ADAPT iteration.
    """

    def __init__(self,
                 nucleus: QuasiparticleNucleus,
                 ref_state: np.ndarray) -> None:
        """
        Initialization of the ADAPTAnsatz object.

        Args:
            nucleus (Nucleus): Nucleus object.
            ref_state (np.ndarray): Reference state of the ansatz.
            pool_format (str): Format of the operator pool.
            operators_list (list): List of operators to be used in the ansatz (optional).
        """
        super().__init__(nucleus, ref_state)
        self.added_operators = []
        self.minimum = False
        
        self.energy_calls=0

    def build_ansatz(self, parameters: list) -> np.ndarray:
        """
        Returns the state of the ansatz on a given VQE iteratioin, after building it with the given paramters and the operators in the pool.

        Args:
            parameters (list): Values of the parameters of a given VQE iteration.

        Returns:
            np.ndarray: Ansatz state.        
        """
        ansatz=self.ref_state

        for i,op in enumerate(self.added_operators):
            ansatz = expm_multiply(-parameters[i]*op.op_matrix, ansatz, traceA = 0.0)
        return ansatz

    def energy(self, parameters: list) -> float:
        """
        Returns the energy of the ansatz on a given VQE iteration.

        Args:
            parameters (list): Values of the parameters of a given VQE iteration.
        
        Returns:
            float: Energy of the ansatz.
        """
        
        self.energy_calls+=1
        
        if len(parameters) != 0:
            if self.count_fcalls == True:
                self.fcalls += 1
            new_ansatz = self.build_ansatz(parameters)
            E = new_ansatz.conj().T.dot(self.nucleus.H.dot(new_ansatz))
            self.last_energy = E
            return E
        else:
            E = self.ansatz.conj().T.dot(self.nucleus.H.dot(self.ansatz))
            self.last_energy = E
            return E

    def choose_operator(self) -> tuple:
        """
        Returns the next operator and its gradient, after an ADAPT iteration.

        Returns:
            TwoBodyExcitationOperator: Next operator.
            float: Gradient of the next operator.
        """
        self.ansatz = self.build_ansatz(self.parameters)
        gradients = []
        psi = self.ansatz
        
        self.energy_calls+=4*len(self.operator_pool)
        
        gradients = [psi.conj().T @ (op.commutator @ psi).real for op in self.operator_pool]
        
        max_gradient = max(gradients, key=abs)
        
        max_operator = self.operator_pool[gradients.index(max_gradient)]
        
        return max_operator,max_gradient

class QP_QuantumADAPTAnsatz(Ansatz):
    """
    Child Ansatz class to define the ADAPT ansatz for VQE.

    Attributes:
        nucleus (Nucleus): Nucleus object.
        ref_state (np.ndarray): Reference state of the ansatz.
        pool_format (str): Format of the operator pool.
        operators_list (list): List of operators to be used in the ansatz.
        added_operators (list): List of operators added to the ansatz.
        minimum (bool): If True, the ansatz has reached the minimum energy.
        E0 (float): Energy of the ansatz without any excitation operators.

    Methods:
        build_ansatz: Returns the state of the ansatz on a given VQE iteratioin, after building it with the given paramters and the operators in the pool.
        energy: Returns the energy of the ansatz on a given VQE iteration.
        choose_operator: Returns the next operator and its gradient, after an ADAPT iteration.
    """

    def __init__(self,
                 nucleus : QuasiparticleNucleus,
                 ref_state : np.ndarray,
                 exact : bool,
                 nshots : int = 1000,
                 max_executions:int = 1000) -> None:
        """
        Initialization of the ADAPTAnsatz object.

        Args:
            nucleus (Nucleus): Nucleus object.
            ref_state (np.ndarray): Reference state of the ansatz.
            pool_format (str): Format of the operator pool.
            operators_list (list): List of operators to be used in the ansatz (optional).
        """
        super().__init__(nucleus, ref_state)
        self.nucleus = nucleus
        self.added_operators = []
        self.minimum = False
        self.exact = exact
        self.nshots = nshots
        
        self.max_executions=max_executions
        self.f = io.StringIO()
        self.energy_calls=0
        
        self.composer = QP_Circuits_Composser(nucleus=self.nucleus.name,
                                            shell=self.nucleus.shell,
                                            ref_state=self.ref_state,
                                            parameters=[],
                                            operators_used=[],
                                            exact=self.exact,
                                            nshots=self.nshots)        

    def energy(self, parameters: list) -> float:
        """
        Returns the energy of the ansatz on a given VQE iteration.

        Args:
            parameters (list): Values of the parameters of a given VQE iteration.
        
        Returns:
            float: Energy of the ansatz.
        """
         
        self.energy_calls+=1
        
        if self.energy_calls>self.max_executions:
            self.energy_calls = self.max_executions
            raise RuntimeError
        
        self.composer.parameters=parameters
        self.composer.operators_used=self.added_operators
        
        with contextlib.redirect_stdout(self.f):
            E = self.composer.Qibo_measure_Energy()[0]
            
        self.last_energy = E
        
        return E

    def choose_operator(self) -> tuple:
        """
        Returns the next operator and its gradient, after an ADAPT iteration.

        Returns:
            TwoBodyExcitationOperator: Next operator.
            float: Gradient of the next operator.
        """
        self.energy_calls += 4*len(self.nucleus.ops_hop)
        
        with contextlib.redirect_stdout(self.f):
            gradients = [(self.composer.Qibo_measure_gradient((op.A, op.B))) for op in self.operator_pool]
        max_gradient = max(gradients, key=abs)
        max_operator = self.operator_pool[gradients.index(max_gradient)]
        
        return max_operator, max_gradient
    
class QP_ADAPT_mixed_Ansatz(Ansatz):
    """
    Child Ansatz class to define the ADAPT ansatz for VQE.

    Attributes:
        nucleus (Nucleus): Nucleus object.
        ref_state (np.ndarray): Reference state of the ansatz.
        pool_format (str): Format of the operator pool.
        operators_list (list): List of operators to be used in the ansatz.
        added_operators (list): List of operators added to the ansatz.
        minimum (bool): If True, the ansatz has reached the minimum energy.
        E0 (float): Energy of the ansatz without any excitation operators.

    Methods:
        build_ansatz: Returns the state of the ansatz on a given VQE iteratioin, after building it with the given paramters and the operators in the pool.
        energy: Returns the energy of the ansatz on a given VQE iteration.
        choose_operator: Returns the next operator and its gradient, after an ADAPT iteration.
    """

    def __init__(self,
                 nucleus: QuasiparticleNucleus,
                 ref_state: np.ndarray, 
                 data: dict,
                 exact:bool=True,
                 nshots:int = 1000,
                 max_executions:int = 100) -> None:
        """
        Initialization of the ADAPTAnsatz object.

        Args:
            nucleus (Nucleus): Nucleus object.
            ref_state (np.ndarray): Reference state of the ansatz.
            pool_format (str): Format of the operator pool.
            operators_list (list): List of operators to be used in the ansatz (optional).
        """
        super().__init__(nucleus, ref_state)
        self.added_operators = []
        self.minimum = False
        self.data=data
        self.exact=exact
        self.nshots=nshots
        self.max_executions=max_executions
        self.f = io.StringIO()
        self.energy_calls=0
        self.ref_state = ref_state
        
        self.composer = QP_Circuits_Composser(nucleus = self.nucleus.name,
                                  shell=self.nucleus.shell,
                                  ref_state = self.ref_state,
                                  parameters = [],
                                  operators_used = self.added_operators,
                                  exact = self.exact,
                                  nshots = self.nshots)
        
        self.capas=0

    def energy(self, parameters) -> float:
        """
        Returns the energy of the ansatz on a given VQE iteration.

        Args:
            parameters (list): Values of the parameters of a given VQE iteration.
        
        Returns:
            float: Energy of the ansatz.
        """

        self.energy_calls+=1
        
        if self.energy_calls>self.max_executions:
            self.energy_calls = self.max_executions
            raise RuntimeError
        
        self.composer.parameters = parameters
        self.composer.operators_used = self.added_operators
        
        
        with contextlib.redirect_stdout(self.f):
            E = self.composer.Qibo_measure_Energy()[0]
            
        self.last_energy = E
        
        return E

    def choose_operator(self):
        i=self.capas
        return self.data['used_operators'][i]
    
class QP_Roto_ADAPT_Ansatz(Ansatz):
    """
    Child Ansatz class to define the ADAPT ansatz for VQE.

    Attributes:
        nucleus (Nucleus): Nucleus object.
        ref_state (np.ndarray): Reference state of the ansatz.
        pool_format (str): Format of the operator pool.
        operators_list (list): List of operators to be used in the ansatz.
        added_operators (list): List of operators added to the ansatz.
        minimum (bool): If True, the ansatz has reached the minimum energy.
        E0 (float): Energy of the ansatz without any excitation operators.

    Methods:
        build_ansatz: Returns the state of the ansatz on a given VQE iteratioin, after building it with the given paramters and the operators in the pool.
        energy: Returns the energy of the ansatz on a given VQE iteration.
        choose_operator: Returns the next operator and its gradient, after an ADAPT iteration.
    """

    def __init__(self,
                 nucleus: QuasiparticleNucleus,
                 ref_state: np.ndarray) -> None:
        """
        Initialization of the ADAPTAnsatz object.

        Args:
            nucleus (Nucleus): Nucleus object.
            ref_state (np.ndarray): Reference state of the ansatz.
            pool_format (str): Format of the operator pool.
            operators_list (list): List of operators to be used in the ansatz (optional).
        """
        super().__init__(nucleus, ref_state)
        self.added_operators = []
        self.minimum = False
        
        self.energy_calls=0
        self.global_best_energy = float('inf')
        self.global_best_params = []

    def build_ansatz(self, parameters: list) -> np.ndarray:
        """
        Returns the state of the ansatz on a given VQE iteratioin, after building it with the given paramters and the operators in the pool.

        Args:
            parameters (list): Values of the parameters of a given VQE iteration.

        Returns:
            np.ndarray: Ansatz state.        
        """
        ansatz=self.ref_state

        for i,op in enumerate(self.added_operators):
            ansatz = expm_multiply(-parameters[i]*op.op_matrix, ansatz, traceA = 0.0)
        return ansatz

    def energy(self, parameters: list, save_ansatz=False) -> float:
        """
        Returns the energy of the ansatz on a given VQE iteration.

        Args:
            parameters (list): Values of the parameters of a given VQE iteration.
        
        Returns:
            float: Energy of the ansatz.
        """
        
        
        self.energy_calls+=1
        
        if len(parameters) != 0:
            if self.count_fcalls == True:
                self.fcalls += 1
            new_ansatz = self.build_ansatz(parameters)
            E = new_ansatz.conj().T.dot(self.nucleus.H.dot(new_ansatz))
            self.last_energy = E
            
            if E < self.global_best_energy:
                self.global_best_energy = E
                self.global_best_params = parameters
            return E
        else:
            E = self.ansatz.conj().T.dot(self.nucleus.H.dot(self.ansatz))
            self.last_energy = E
            if E < self.global_best_energy:
                self.global_best_energy = E
                self.global_best_params = parameters
            return E    
        
    def find_min(self, op):
        
        self.added_operators.append(op)
        
        parameters = self.parameters
        
        self.energy_calls-=1
        d = 2 * np.pi / 5.0

        cd = np.cos(d)
        sd = np.sin(d)
        c2d = np.cos(2*d)
        s2d = np.sin(2*d)

        E0 = self.energy(parameters+[0])

        E_p1 = self.energy(parameters+[d])

        E_m1 = self.energy(parameters+[-d])

        E_p2 = self.energy(parameters+[2*d])

        E_m2 = self.energy(parameters+[-2*d])

        self.added_operators.pop()

        # 2. Cálculo Analítico de Coeficientes (DFT Directa)
        # Al ser equiespaciados, los coeficientes son proyecciones ortogonales.
        # Factor de normalización 2/N (excepto c0 que es 1/N)
        norm = 2.0 / 5.0

        # Sumas auxiliares para aprovechar la simetría par/impar
        sum_all = E0 + E_p1 + E_m1 + E_p2 + E_m2
        diff_1 = E_p1 - E_m1
        sum_1  = E_p1 + E_m1
        diff_2 = E_p2 - E_m2
        sum_2  = E_p2 + E_m2

        c0 = sum_all / 5.0

        # Proyección para frecuencia 1
        c1 = norm * (E0 + sum_1 * cd + sum_2 * c2d)
        s1 = norm * (diff_1 * sd + diff_2 * s2d)

        # Proyección para frecuencia 2
        # Nota matemática: cos(4d) = cos(d) y sin(4d) = -sin(d)
        c2 = norm * (E0 + sum_1 * c2d + sum_2 * cd)
        s2 = norm * (diff_1 * s2d - diff_2 * sd)
        
        
        def objective_function(theta_array):
            theta = theta_array[0] # L-BFGS espera arrays
            
            # Pre-calculamos trigonométricas para eficiencia
            ct = np.cos(theta)
            st = np.sin(theta)
            c2t = np.cos(2*theta) # O usar 2*ct*ct - 1
            s2t = np.sin(2*theta) # O usar 2*st*ct
            
            # Valor de la Energía
            E_val = (c0 + 
                    c1 * ct + s1 * st + 
                    c2 * c2t + s2 * s2t)
            
            # Valor del Gradiente (Jacobiano)
            # d/dt (cos t) = -sin t, etc.
            grad_val = (-c1 * st + s1 * ct 
                        -2 * c2 * s2t + 2 * s2 * c2t)
            
            return E_val, np.array([grad_val])
        
        grid = np.linspace(-np.pi, np.pi, 20) 
        # Evaluamos solo la energía en el grid (vectorizado es muy rápido)
        
        vals = (c0 + c1*np.cos(grid) + s1*np.sin(grid) + c2*np.cos(2*grid) + s2*np.sin(2*grid))
        
        initial_guess = [grid[np.argmin(vals)]]
        
        # Optimización L-BFGS-B con Jacobiano
        # bounds=[-pi, pi] ayuda a mantener la búsqueda acotada
        res = minimize(
            objective_function, 
            initial_guess, 
            method='L-BFGS-B', 
            jac=True,  # Indica que objective_function devuelve (val, grad)
            bounds=[(-np.pi, np.pi)],
            tol=1e-12
        )
        
        return res.x[0], res.fun     

    def choose_operator(self) -> tuple:
        """
        Returns the next operator and its gradient, after an ADAPT iteration.

        Returns:
            TwoBodyExcitationOperator: Next operator.
            float: Gradient of the next operator.
        """
        
        min_energies = []
        best_parameters = []
        
        for op in self.operator_pool:
            par, en = self.find_min(op)
            
            best_parameters.append(par)
            min_energies.append(en)
            
        min_energy = min(min_energies)
        
        best_operator = self.operator_pool[min_energies.index(min_energy)]
        best_parameter = best_parameters[min_energies.index(min_energy)]
        
        return best_operator, min_energy, best_parameter
    
    def set_optimized_state(self, parameters: list, energy: float) -> None:
        """
        Actualiza el estado interno del ansatz con los parámetros y la energía 
        óptimos encontrados por el optimizador, sin realizar nuevas mediciones.
        """
        # Actualizamos parámetros cacheados
        
        self.parameters = parameters
        
        # INYECCIÓN DIRECTA DE ENERGÍA (¡Sin coste!)
        self.last_energy = energy
           
class QP_Quantum_Roto_ADAPTAnsatz(Ansatz):
    """
    Child Ansatz class to define the ADAPT ansatz for VQE.

    Attributes:
        nucleus (Nucleus): Nucleus object.
        ref_state (np.ndarray): Reference state of the ansatz.
        pool_format (str): Format of the operator pool.
        operators_list (list): List of operators to be used in the ansatz.
        added_operators (list): List of operators added to the ansatz.
        minimum (bool): If True, the ansatz has reached the minimum energy.
        E0 (float): Energy of the ansatz without any excitation operators.

    Methods:
        build_ansatz: Returns the state of the ansatz on a given VQE iteratioin, after building it with the given paramters and the operators in the pool.
        energy: Returns the energy of the ansatz on a given VQE iteration.
        choose_operator: Returns the next operator and its gradient, after an ADAPT iteration.
    """

    def __init__(self,
                 nucleus : QuasiparticleNucleus,
                 ref_state : np.ndarray,
                 exact : bool,
                 nshots : int = 1000,
                 max_executions:int = 1000) -> None:
        """
        Initialization of the ADAPTAnsatz object.

        Args:
            nucleus (Nucleus): Nucleus object.
            ref_state (np.ndarray): Reference state of the ansatz.
            pool_format (str): Format of the operator pool.
            operators_list (list): List of operators to be used in the ansatz (optional).
        """
        super().__init__(nucleus, ref_state)
        self.nucleus = nucleus
        self.added_operators = []
        self.minimum = False
        self.exact = exact
        self.nshots = nshots
        
        self.max_executions=max_executions
        self.f = io.StringIO()
        self.energy_calls=0
        self.global_best_energy = float('inf')
        self.global_best_params = []
        
        self.composer = QP_Circuits_Composser(nucleus=self.nucleus.name,
                                            shell=self.nucleus.shell,
                                            ref_state=self.ref_state,
                                            parameters=[],
                                            operators_used=[],
                                            exact=self.exact,
                                            nshots=self.nshots)        

        
        
    def energy(self, parameters: list, save_ansatz = False) -> float:
        """
        Returns the energy of the ansatz on a given VQE iteration.

        Args:
            parameters (list): Values of the parameters of a given VQE iteration.
        
        Returns:
            float: Energy of the ansatz.
        """
        
        
        self.energy_calls+=1
        
        if self.energy_calls>self.max_executions:
            self.energy_calls = self.max_executions
            raise RuntimeError
        
        self.composer.parameters=parameters
        self.composer.operators_used=self.added_operators
        
        with contextlib.redirect_stdout(self.f):
            energy_meas = self.composer.Qibo_measure_Energy()
            E = energy_meas[0]
        
        if E < self.global_best_energy:
                self.global_best_energy = E
                self.global_best_params = parameters
        
        self.last_energy = E
        
        self.cached_parameters = parameters
        
        if save_ansatz:
            self.ansatz = energy_meas[-1] 
        
        return E

    def choose_operator(self) -> tuple:
        """
        Returns the next operator and its gradient, after an ADAPT iteration.

        Returns:
            TwoBodyExcitationOperator: Next operator.
            float: Gradient of the next operator.
        """
        
        min_energies = []
        best_parameters = []
        
        for op in self.operator_pool:
            if ((len(self.added_operators)==0 or (op.A != self.added_operators[-1].A or op.B != self.added_operators[-1].B)) 
                and not (self.ansatz.probabilities([op.A, op.B])[0] > 0.99999 or self.ansatz.probabilities([op.A, op.B])[3] > 0.99999)):
                self.energy_calls += 4
                par, en = self.composer.Qibo_find_min([op.A, op.B])
                best_parameters.append(par)
                min_energies.append(en)
            else:
                best_parameters.append(0)
                min_energies.append(100)
                
            
        min_energy = min(min_energies)
        self.last_energy = min_energy
        self.composer.E0 = min_energy
        
        best_operator = self.operator_pool[min_energies.index(min_energy)]
        best_parameter = best_parameters[min_energies.index(min_energy)]
        
        
        
        return best_operator, min_energy, best_parameter
    
    def set_optimized_state(self, parameters: list, energy: float) -> None:
        """
        Actualiza el estado interno del ansatz con los parámetros y la energía 
        óptimos encontrados por el optimizador, sin realizar nuevas mediciones.
        """
        # Actualizamos parámetros cacheados
        self.cached_parameters = parameters
        self.composer.parameters = parameters
        self.composer.operators_used = self.added_operators
        
        # INYECCIÓN DIRECTA DE ENERGÍA (¡Sin coste!)
        self.last_energy = energy
        self.composer.E0 = energy
        
class QP_Roto_Ansatz(Ansatz):
    """
    Child Ansatz class to define the ADAPT ansatz for VQE.

    Attributes:
        nucleus (Nucleus): Nucleus object.
        ref_state (np.ndarray): Reference state of the ansatz.
        pool_format (str): Format of the operator pool.
        operators_list (list): List of operators to be used in the ansatz.
        added_operators (list): List of operators added to the ansatz.
        minimum (bool): If True, the ansatz has reached the minimum energy.
        E0 (float): Energy of the ansatz without any excitation operators.

    Methods:
        build_ansatz: Returns the state of the ansatz on a given VQE iteratioin, after building it with the given paramters and the operators in the pool.
        energy: Returns the energy of the ansatz on a given VQE iteration.
        choose_operator: Returns the next operator and its gradient, after an ADAPT iteration.
    """

    def __init__(self,
                 nucleus: QuasiparticleNucleus,
                 ref_state: np.ndarray) -> None:
        """
        Initialization of the ADAPTAnsatz object.

        Args:
            nucleus (Nucleus): Nucleus object.
            ref_state (np.ndarray): Reference state of the ansatz.
            pool_format (str): Format of the operator pool.
            operators_list (list): List of operators to be used in the ansatz (optional).
        """
        super().__init__(nucleus, ref_state)
        self.added_operators = []
        self.minimum = False
        
        self.energy_calls=0

    def build_ansatz(self, parameters: list) -> np.ndarray:
        """
        Returns the state of the ansatz on a given VQE iteratioin, after building it with the given paramters and the operators in the pool.

        Args:
            parameters (list): Values of the parameters of a given VQE iteration.

        Returns:
            np.ndarray: Ansatz state.        
        """
        ansatz=self.ref_state

        for i,op in enumerate(self.added_operators):
            ansatz = expm_multiply(-parameters[i]*op.op_matrix, ansatz, traceA = 0.0)
        return ansatz

    def energy(self, parameters: list, save_ansatz = False) -> float:
        """
        Returns the energy of the ansatz on a given VQE iteration.

        Args:
            parameters (list): Values of the parameters of a given VQE iteration.
        
        Returns:
            float: Energy of the ansatz.
        """
        
        
        self.energy_calls+=1
        
        if len(parameters) != 0:
            if self.count_fcalls == True:
                self.fcalls += 1
            new_ansatz = self.build_ansatz(parameters)
            E = new_ansatz.conj().T.dot(self.nucleus.H.dot(new_ansatz))
        
            return E
        else:
            E = self.ansatz.conj().T.dot(self.nucleus.H.dot(self.ansatz))
           
            return E    
        
    def find_min(self, op):
        
        self.added_operators.append(op)
        
        parameters = self.parameters
        
        self.energy_calls -= 1
        
        E0 = self.energy(parameters+[0])
        
        E_p2 = self.energy(parameters+[np.pi/2])
            
        E_p4 = self.energy(parameters+[np.pi/4])
            
        E_m2 = self.energy(parameters+[-np.pi/2])
            
        E_m4 = self.energy(parameters+[-np.pi/4])
        
        self.added_operators.pop()
        
        # E(t) = c0 + c1*cos(t) + s1*sin(t) + c2*cos(2t) + s2*sin(2t)
    
        # s1 viene directo del rango amplio
        s1 = (E_p2 - E_m2) / 2.0
        
        # E(pi/4) - E(-pi/4) = sqrt(2)*s1 + 2*s2
        s2 = ((E_p4 - E_m4) / 2.0) - (s1 * np.sqrt(0.5))
        
        A = (E_p2 + E_m2) / 2.0   # A = c0 - c2
        B = (E_p4 + E_m4) / 2.0   # B = c0 + c1/sqrt(2)
        #  E0 = c0 + c1 + c2
        
        denom = 2 - np.sqrt(2)
        c0 = (E0 + A - np.sqrt(2)*B) / denom
        c1 = np.sqrt(2) * (B - c0)
        c2 = c0 - A
        
        #print(op.A, op.B, c0, c1, s1, c2, s2)
        
        def objective_function(theta_array):
            theta = theta_array[0] # L-BFGS espera arrays
            
            # Pre-calculamos trigonométricas para eficiencia
            ct = np.cos(theta)
            st = np.sin(theta)
            c2t = np.cos(2*theta) # O usar 2*ct*ct - 1
            s2t = np.sin(2*theta) # O usar 2*st*ct
            
            # Valor de la Energía
            E_val = (c0 + 
                    c1 * ct + s1 * st + 
                    c2 * c2t + s2 * s2t)
            
            # Valor del Gradiente (Jacobiano)
            # d/dt (cos t) = -sin t, etc.
            grad_val = (-c1 * st + s1 * ct 
                        -2 * c2 * s2t + 2 * s2 * c2t)
            
            return E_val, np.array([grad_val])
        
        grid = np.linspace(-np.pi, np.pi, 20) 
        # Evaluamos solo la energía en el grid (vectorizado es muy rápido)
        
        vals = (c0 + c1*np.cos(grid) + s1*np.sin(grid) + c2*np.cos(2*grid) + s2*np.sin(2*grid))
        
        initial_guess = [grid[np.argmin(vals)]]
        
        # Optimización L-BFGS-B con Jacobiano
        # bounds=[-pi, pi] ayuda a mantener la búsqueda acotada
        res = minimize(
            objective_function, 
            initial_guess, 
            method='L-BFGS-B', 
            jac=True,  # Indica que objective_function devuelve (val, grad)
            bounds=[(-np.pi, np.pi)],
            tol=1e-12
        )
        
        return res.x[0], res.fun     

    def choose_operator(self) -> tuple:
        """
        Returns the next operator and its gradient, after an ADAPT iteration.

        Returns:
            TwoBodyExcitationOperator: Next operator.
            float: Gradient of the next operator.
        """
        
        min_energies = []
        best_parameters = []
        
        for op in self.operator_pool:
            if not self.added_operators or not (op.A == self.added_operators[-1].A and op.B == self.added_operators[-1].B):
                par, en = self.find_min(op)
                
                best_parameters.append(par)
                min_energies.append(en)
            else:
                best_parameters.append(0)
                min_energies.append(100)
            
        min_energy = min(min_energies)
        
        best_operator = self.operator_pool[min_energies.index(min_energy)]
        best_parameter = best_parameters[min_energies.index(min_energy)]
        
        return best_operator, min_energy, best_parameter
     
    def set_optimized_state(self, parameters: list, energy: float) -> None:
        """
        Actualiza el estado interno del ansatz con los parámetros y la energía 
        óptimos encontrados por el optimizador, sin realizar nuevas mediciones.
        """
        # Actualizamos parámetros cacheados

        self.parameters = parameters
        
    
            
class QP_Quantum_Roto_Ansatz(Ansatz):
    """
    Child Ansatz class to define the ADAPT ansatz for VQE.

    Attributes:
        nucleus (Nucleus): Nucleus object.
        ref_state (np.ndarray): Reference state of the ansatz.
        pool_format (str): Format of the operator pool.
        operators_list (list): List of operators to be used in the ansatz.
        added_operators (list): List of operators added to the ansatz.
        minimum (bool): If True, the ansatz has reached the minimum energy.
        E0 (float): Energy of the ansatz without any excitation operators.

    Methods:
        build_ansatz: Returns the state of the ansatz on a given VQE iteratioin, after building it with the given paramters and the operators in the pool.
        energy: Returns the energy of the ansatz on a given VQE iteration.
        choose_operator: Returns the next operator and its gradient, after an ADAPT iteration.
    """

    def __init__(self,
                 nucleus : QuasiparticleNucleus,
                 ref_state : np.ndarray,
                 exact : bool,
                 nshots : int = 1000,
                 max_executions:int = 1000) -> None:
        """
        Initialization of the ADAPTAnsatz object.

        Args:
            nucleus (Nucleus): Nucleus object.
            ref_state (np.ndarray): Reference state of the ansatz.
            pool_format (str): Format of the operator pool.
            operators_list (list): List of operators to be used in the ansatz (optional).
        """
        super().__init__(nucleus, ref_state)
        self.nucleus = nucleus
        self.added_operators = []
        self.minimum = False
        self.exact = exact
        self.nshots = nshots
        
        self.max_executions=max_executions
        self.f = io.StringIO()
        self.energy_calls=0
        
        self.composer = QP_Circuits_Composser(nucleus=self.nucleus.name,
                                            shell=self.nucleus.shell,
                                            ref_state=self.ref_state,
                                            parameters=[],
                                            operators_used=[],
                                            exact=self.exact,
                                            nshots=self.nshots)        

        
        
    def energy(self, parameters: list, save_ansatz = False) -> float:
        """
        Returns the energy of the ansatz on a given VQE iteration.

        Args:
            parameters (list): Values of the parameters of a given VQE iteration.
        
        Returns:
            float: Energy of the ansatz.
        """
        
        
        self.energy_calls+=1
        
        if self.energy_calls>self.max_executions:
            self.energy_calls = self.max_executions
            raise RuntimeError
        
        self.composer.parameters = parameters 
        self.composer.operators_used = self.added_operators
        
      
        
        with contextlib.redirect_stdout(self.f):
            energy_meas = self.composer.Qibo_measure_Energy()
            E = energy_meas[0]
            
            if save_ansatz:
                self.ansatz = energy_meas[-1] 
            
        self.last_energy = E
        
        self.cached_parameters = parameters
        
        return E


    def set_optimized_state(self, parameters: list, energy: float) -> None:
        """
        Actualiza el estado interno del ansatz con los parámetros y la energía 
        óptimos encontrados por el optimizador, sin realizar nuevas mediciones.
        """
        # Actualizamos parámetros cacheados

        self.composer.parameters = parameters
        self.composer.operators_used = self.added_operators
        

        self.composer.E0 = energy
        
        
    def choose_operator(self) -> tuple:
        """
        Returns the next operator and its gradient, after an ADAPT iteration.

        Returns:
            TwoBodyExcitationOperator: Next operator.
            float: Gradient of the next operator.
        """
        
        
        min_energies = []
        best_parameters = []
        
        for op in self.operator_pool:
            if ((len(self.added_operators)==0 or (op.A != self.added_operators[-1].A or op.B != self.added_operators[-1].B)) 
                and not (self.ansatz.probabilities([op.A, op.B])[0] > 0.999 or self.ansatz.probabilities([op.A, op.B])[3] > 0.999)):
            
                self.energy_calls += 4
                par, en = self.composer.Qibo_find_min([op.A, op.B])
                best_parameters.append(par)
                min_energies.append(en)
                
            else:              
                
                best_parameters.append(0)
                min_energies.append(100)
            
        min_energy = min(min_energies)
        self.last_energy = min_energy
        self.composer.E0 = min_energy
        
        best_operator = self.operator_pool[min_energies.index(min_energy)]
        best_parameter = best_parameters[min_energies.index(min_energy)]
        
        
        return best_operator, min_energy, best_parameter


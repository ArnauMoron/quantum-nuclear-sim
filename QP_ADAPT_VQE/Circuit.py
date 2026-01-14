import numpy as np

from qibo import gates
from qibo.models.circuit import Circuit
from qibo.result import MeasurementOutcomes

from QP_ADAPT_VQE.Nucleus import QuasiparticleNucleus

from scipy.optimize import minimize
import contextlib
import io

class QP_Circuits_Composser():


    
    def __init__(self,
                 nucleus:str='Be6',
                 n_qubits:int = 6,
                 ref_state:int = 0,
                 parameters:list=[],
                 operators_used: list=[],
                 exact:bool=True,
                 nshots:int=1000):
        
        self.exact = exact
        self.nshots = nshots
        nuc = QuasiparticleNucleus(nucleus, n_qubits=n_qubits)
        self.nuc = nuc
        
        self.operator_pool = nuc.ops_hop
        
        self.n_qubits = n_qubits//2
        
        self.operators_used = operators_used
        self.parameters = parameters
        self.ref_state = ref_state

        self.ops_1_body = nuc.ops_1_body
        self.ops_2_body = nuc.ops_2_body
        self.ops_hop = nuc.ops_hop
        
        self.f = io.StringIO()
        
        
            
    def Qibo_ref_state_composer(self):
        
        ref_state_indexes = self.nuc.qp_states[self.ref_state]

        circuit = Circuit(self.n_qubits)

        for index in ref_state_indexes:
            circuit.add(gates.X(index))

        for q in range(self.n_qubits):
            if q not in ref_state_indexes:
                circuit.add(gates.I(q))

        return circuit

    def Qibo_layer_composer(self):
        circ = Circuit(self.n_qubits)
        
        for i in range(len(self.operators_used)):
            A = self.operators_used[i].A
            B = self.operators_used[i].B
            par = self.parameters[i]
            circ.add(gates.GIVENS(A, B, -par))
            
        return circ
    
    def Qibo_all_circuits(self, circ = None):

        if circ == None:
            main_circuit = self.Qibo_ref_state_composer() + self.Qibo_layer_composer()  
        else: 
            main_circuit = circ
            
        XX_observables_circ = main_circuit.copy()
        
        YY_observables_circ = main_circuit.copy()

        for q in range(self.n_qubits):
            XX_observables_circ.add(gates.H(q))
            YY_observables_circ.add(gates.RX(q, np.pi / 2))
    
        return [main_circuit, XX_observables_circ, YY_observables_circ]
        
    def Qibo_measure_Energy(self, circuits=None, save_energy = True):
        
        if circuits == None:
            Qibo_circs = self.Qibo_all_circuits()
        else:
            Qibo_circs=circuits
            
    
        E_diag = 0
        
        E_hop = 0
    
        main_circ = Qibo_circs[0].copy()
        circ_x = Qibo_circs[1].copy()
        circ_y = Qibo_circs[2].copy()
        
        if not self.exact:
            
            main_circ.add([gates.M(i) for i in range(main_circ.nqubits)])
            circ_x.add([gates.M(i) for i in range(main_circ.nqubits)])
            circ_y.add([gates.M(i) for i in range(main_circ.nqubits)])
            
            result = main_circ.execute(nshots=self.nshots)
            result_X = circ_x.execute(nshots=self.nshots)
            result_Y = circ_y.execute(nshots=self.nshots)
     
            result = MeasurementOutcomes(measurements=result.measurements, backend=result.backend, probabilities=None, samples=result.samples(), nshots=self.nshots)  
            result_X = MeasurementOutcomes(measurements=result.measurements, backend=result.backend, probabilities=None, samples=result_X.samples(), nshots=self.nshots) 
            result_Y = MeasurementOutcomes(measurements=result.measurements, backend=result.backend, probabilities=None, samples=result_Y.samples(), nshots=self.nshots) 
            
        else:
            
            result = main_circ()
            result_X = circ_x()
            result_Y = circ_y() 
            print(result)
            
        for op in self.ops_1_body:
            E_diag += op.g*result.probabilities([op.A])[1]
        
        for op in self.ops_2_body:
            E_diag += op.g*result.probabilities([op.A, op.B])[3]
        
        
        
        
        for op in self.ops_hop:
            prob_x = result_X.probabilities([op.A, op.B])
            prob_y = result_Y.probabilities([op.A, op.B])
            
            E_x = prob_x[0] - prob_x[1] - prob_x[2] + prob_x[3]
            E_y = prob_y[0] - prob_y[1] - prob_y[2] + prob_y[3]

            E_hop += 0.5*op.g*(E_x + E_y)
        
        E_t = E_diag + E_hop   
        
        if save_energy:
            self.E0 = E_t
        
        return E_t, E_diag, E_hop

    def Qibo_measure_gradient(self, op_index):
        A = op_index[0]
        B = op_index[1]
        
        C2 = (1-np.sqrt(2))/2
        
        circs_plus, circs_plus_plus, circs_min, circs_min_min = self.Qibo_gradient_circuits(A,B)
        
        gradient = (self.Qibo_measure_Energy(circs_plus)[0] - self.Qibo_measure_Energy(circs_min)[0]) + C2 * (self.Qibo_measure_Energy(circs_plus_plus)[0] - self.Qibo_measure_Energy(circs_min_min)[0]) 
        
        print(gradient)
        
        return gradient
         
    def Qibo_gradient_circuits(self, A, B):
        
        main_circ = self.Qibo_ref_state_composer() + self.Qibo_layer_composer() 
        
        circ_plus = main_circ.copy()
        circ_min = main_circ.copy()
        circ_plus_plus = main_circ.copy()
        circ_min_min = main_circ.copy()
        
        d = 0.4 * np.pi
        
        circ_plus.add(gates.GIVENS(A, B, -d))
        circ_min.add(gates.GIVENS(A, B, d))
        circ_plus_plus.add(gates.GIVENS(A, B, -2*d))
        circ_min_min.add(gates.GIVENS(A, B, 2*d))
        
        
        circs_plus = self.Qibo_all_circuits(circ_plus)
        circs_min = self.Qibo_all_circuits(circ_min)
        circs_plus_plus = self.Qibo_all_circuits(circ_plus_plus)
        circs_min_min = self.Qibo_all_circuits(circ_min_min)
        
        return circs_plus, circs_plus_plus, circs_min, circs_min_min
        
    def Qibo_find_min(self, op_index):
        A = op_index[0]
        B = op_index[1]
        
        d = 0.4 * np.pi 
        cd = np.cos(d)
        sd = np.sin(d)
        c2d = np.cos(2*d)
        s2d = np.sin(2*d)
        
        circs_plus, circs_plus_plus, circs_min, circs_min_min = self.Qibo_gradient_circuits(A,B)
        
        with contextlib.redirect_stdout(self.f):
            
            E0 = self.E0
            
            E_p2 = self.Qibo_measure_Energy(circs_plus_plus, False)[0]
            
            E_p1 = self.Qibo_measure_Energy(circs_plus, False)[0]
            
            E_m2 = self.Qibo_measure_Energy(circs_min_min, False)[0]
            
            E_m1 = self.Qibo_measure_Energy(circs_min, False)[0]
        
        # E(t) = c0 + c1*cos(t) + s1*sin(t) + c2*cos(2t) + s2*sin(2t)
    
        sum_1 = E_p1 + E_m1
        dif_1 = E_p1 - E_m1
        sum_2 = E_p2 + E_m2
        dif_2 = E_p2 - E_m2
        
        # Factor de normalización de Fourier (2/N)
        norm = 0.4 
        
        # Componente DC (Promedio simple)
        c0 = (E0 + sum_1 + sum_2) / 5.0
        
        # Frecuencia fundamental (w)
        c1 = norm * (E0 + sum_1 * cd + sum_2 * c2d)
        s1 = norm * (dif_1 * sd + dif_2 * s2d)
        
        # Segunda frecuencia (2w)
        # Nota: cos(4d) = cos(d) y sin(4d) = -sin(d)
        c2 = norm * (E0 + sum_1 * c2d + sum_2 * cd)
        s2 = norm * (dif_1 * s2d - dif_2 * sd)
        
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
        
        amp_fund = np.sqrt(c1**2 + s1**2)  # Amplitud de la frecuencia pi
        amp_2nd  = np.sqrt(c2**2 + s2**2)  # Amplitud de la frecuencia 2pi

        # 2. Definimos la tolerancia al ruido (ej. 0.1 significa 10%)
        # Si amp_fund es menor al 10% de amp_2nd, asumimos que es ruido o irrelevante.
        epsilon = 0.2 

        if amp_fund < epsilon * amp_2nd or len(self.operators_used)==0:
            grid = np.linspace(-np.pi/2, np.pi/2, 20) 
        else:
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
    
    
        
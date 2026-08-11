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
                 shell:str = 'p',
                 ref_state:int = 0,
                 parameters:list=[],
                 operators_used: list=[],
                 exact:bool=True,
                 nshots:int=1000):
        
        self.exact = exact
        self.nshots = nshots
        nuc = QuasiparticleNucleus(nucleus, shell=shell)
        self.nuc = nuc
        
        self.operators_used = operators_used
        self.parameters = parameters
        self.ref_state = ref_state
        
        self.ref_state_indexes = self.nuc.qp_states[self.ref_state]
        self.operator_pool = nuc.ops_hop
        
        self.n_qubits = len(self.nuc.nucleus.qubits)//2
        
        
        if len(self.ref_state_indexes) != 1 and len(self.ref_state_indexes) != (self.n_qubits - 1):
            self.complex = True
            with open('complejo.dat','a') as o:
                o.write('salta complex')
        else:
            self.complex = False
        

        self.ops_1_body = nuc.ops_1_body
        self.ops_2_body = nuc.ops_2_body
        self.ops_hop = nuc.ops_hop
        
        self.f = io.StringIO()
        
    def Qibo_measure_Energy(self, circ = None):
        
        if self.complex:
            return self.Qibo_measure_Energy_complex(circ)
        else:
            return self.Qibo_measure_Energy_simple(circ)
       
                      
    def Qibo_ref_state_composer(self):

        circuit = Circuit(self.n_qubits)

        for index in self.ref_state_indexes:
            circuit.add(gates.X(index))
            

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
        # YY_observables_circ = main_circuit.copy()

        for q in range(self.n_qubits):
            XX_observables_circ.add(gates.H(q))
            # YY_observables_circ.add(gates.RX(q, np.pi / 2))
    
        return [main_circuit, XX_observables_circ]#, YY_observables_circ]

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
            
            E_p2 = self.Qibo_measure_Energy(circs_plus_plus)[0]
            
            E_p1 = self.Qibo_measure_Energy(circs_plus)[0]
            
            E_m2 = self.Qibo_measure_Energy(circs_min_min)[0]
            
            E_m1 = self.Qibo_measure_Energy(circs_min)[0]
        
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
    
    def Qibo_all_circuits_complex(self, circ = None):
        
        if circ == None:
            main_circuit = self.Qibo_ref_state_composer() + self.Qibo_layer_composer()  
        else: 
            main_circuit = circ
            
        op_circuits = []
        

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
            Qibo_circs=circuits
            
    
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
     
            result = MeasurementOutcomes(measurements=result.measurements, backend=result.backend, probabilities=None, samples=result.samples(), nshots=self.nshots)  
            result_X = MeasurementOutcomes(measurements=result.measurements, backend=result.backend, probabilities=None, samples=result_X.samples(), nshots=self.nshots) 
            # result_Y = MeasurementOutcomes(measurements=result.measurements, backend=result.backend, probabilities=None, samples=result_Y.samples(), nshots=self.nshots) 
            
        else:
            
            result = main_circ()
            result_X = circ_x()
            # result_Y = circ_y() 
            print(result)
            
        for op in self.ops_1_body:
            E_diag += op.g*result.probabilities([op.A])[1]
        
        for op in self.ops_2_body:
            E_diag += op.g*result.probabilities([op.A, op.B])[3]
        
        
        for op in self.ops_hop:
            
            probs_z = result.probabilities([op.A, op.B])
            P_10 = probs_z[2]
            P_01 = probs_z[1]
            
            magnitude_clean = 2 * np.sqrt(P_10 * P_01)

            prob_x = result_X.probabilities([op.A, op.B])
            val_xx = prob_x[0] - prob_x[1] - prob_x[2] + prob_x[3]
            
            # prob_y = result_Y.probabilities([op.A, op.B])
            # val_yy = prob_y[0] - prob_y[1] - prob_y[2] + prob_y[3]
            
            signal = val_xx # + val_yy
            
            if abs(signal) == 0:
                term_energy = 0.0
            else:
                sign_clean = np.sign(signal)
                
                term_energy = op.g * sign_clean * magnitude_clean

            E_hop_rec += term_energy
            
            E_hop += op.g*signal
        
        
        E_t = E_diag + E_hop_rec
        
        return E_t, E_diag, E_hop, result

    def Qibo_measure_Energy_complex(self, circuits=None):
             
        if circuits == None:
            Qibo_circs = self.Qibo_all_circuits_complex()
        else:
            Qibo_circs = circuits
            
        E_diag = 0
        
        E_hop = 0
        
        E_hop_rec = 0
    
        results_x = []
        main_circ = Qibo_circs[0].copy()
        circs_x = Qibo_circs[1].copy()
        circ_global_x = Qibo_circs[2].copy()
        
        if not self.exact:
            
            main_circ.add([gates.M(i) for i in range(main_circ.nqubits)])
            result = main_circ.execute(nshots=self.nshots)
            result = MeasurementOutcomes(measurements=result.measurements, backend=result.backend, probabilities=None, samples=result.samples(), nshots=self.nshots) 
            
            circ_global_x.add([gates.M(i) for i in range(main_circ.nqubits)])
            result_global_x = circ_global_x.execute(nshots=self.nshots)
            result_global_x = MeasurementOutcomes(measurements=result_global_x.measurements, backend=result.backend, probabilities=None, samples=result_global_x.samples(), nshots=self.nshots) 
            
            for circ, op in circs_x:
                circ_x = circ.copy() 
                circ_x.add([gates.M(i) for i in range(main_circ.nqubits)])
                result_X = circ_x.execute(nshots=self.nshots)
                result_X = MeasurementOutcomes(measurements=result.measurements, backend=result.backend, probabilities=None, samples=result_X.samples(), nshots=self.nshots)
                results_x.append((result_X, op))   
             
        else:
            
            result = main_circ()
            result_global_x = circ_global_x()
            
            for circ_x, op in circs_x:
                result_X = circ_x()
                results_x.append((result_X, op))
            
            print(result)
            
        for op in self.ops_1_body:
            E_diag += op.g*result.probabilities([op.A])[1]
        
        for op in self.ops_2_body:
            E_diag += op.g*result.probabilities([op.A, op.B])[3]
        
        
        for res, op in results_x:
            
            indexes_k = [q for q in range(self.n_qubits) if q != op.A and q != op.B]
            indexes = indexes_k + [op.A, op.B]
            
            traced_probs = result.probabilities([op.A, op.B])
            
            if traced_probs[1]*traced_probs[2]!=0:
                
                probs_z = result.probabilities(indexes)
                prob_x = res.probabilities(indexes)
                
                pz_mat = probs_z.reshape(-1, 4)
                px_mat = prob_x.reshape(-1, 4)
                
                magnitudes = 2 * np.sqrt(pz_mat[:, 1] * pz_mat[:, 2])
                signals = px_mat[:, 0] - px_mat[:, 1] - px_mat[:, 2] + px_mat[:, 3]
                signs = np.sign(signals)
                term_energies = op.g * signs * magnitudes
                E_hop_rec += np.sum(term_energies)

                # for k in range(2**(self.n_qubits-2)):
                    
                #     env_in = 4 * k
                #     P_10_k = probs_z[env_in + 2]
                #     P_01_k = probs_z[env_in + 1]
                
                #     magnitude_clean = 2 * np.sqrt(P_10_k * P_01_k)

                #     val_xx = prob_x[env_in + 0] - prob_x[env_in + 1] - prob_x[env_in + 2] + prob_x[env_in + 3]
                
                #     signal = val_xx
                    
                #     if abs(signal) == 0:
                #         term_energy = 0.0
                #     else:
                #         sign_clean = np.sign(signal)
                        
                #         term_energy = op.g * sign_clean * magnitude_clean

                #     E_hop_rec += term_energy
                    
                    
        for op in self.ops_hop:
            
            prob_x = result_global_x.probabilities([op.A, op.B])
            val_xx = prob_x[0] - prob_x[1] - prob_x[2] + prob_x[3]
            
            signal = val_xx
            
            E_hop += op.g*signal
            
    
        
        E_t = E_diag + E_hop_rec
        
        return E_t, E_diag, E_hop, result 
        
        
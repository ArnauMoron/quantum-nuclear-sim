import numpy as np
from scipy.optimize import minimize

from QP_ADAPT_VQE.Nucleus import QuasiparticleNucleus
from QP_ADAPT_VQE.Ansatze import QP_ADAPTAnsatz, QP_ADAPT_mixed_Ansatz, QP_QuantumADAPTAnsatz, QP_Roto_ADAPT_Ansatz, QP_Quantum_Roto_ADAPTAnsatz, QP_Roto_Ansatz, QP_Quantum_Roto_Ansatz


class OptimizationConvergedException(Exception):
    pass

class VQE():
    """
    Parent class to define the Variational Quantum Eigensolvers (VQEs).

    Attributes:
        method (str): Optimization method.
        test_threshold (float): Threshold to stop the optimization.
        energy (list): List of energies.
        rel_error (list): List of relative errors.
        success (bool): If True, the optimization was successful.
        tot_operations (list): List of total operations.
        options (dict): Optimization options.
    
    Methods:
        update_options: Update the optimization options.
    """

    def __init__(self,
                 test_threshold: float = 1e-4,
                 method: str = 'L-BFGS-B',
                 ftol: float = 1e-3,
                 gtol: float = 1e-2,
                 rhobeg: float = 0.2) -> None:
        """
        Initialization of the VQE object.

        Args:
            test_threshold (float): Threshold to stop the optimization.
            method (str): Optimization method.
            ftol (float): Tolerance for the energy.
            gtol (float): Tolerance for the gradient.
            rhoend (float): Tolerance for the constraints.
        """
        self.method = method
        self.test_threshold = test_threshold
       
        self.energy = []
        self.rel_error = []
        self.success = False 
        self.tot_operations = [0]
        try:
            self.method = method
        except method not in ['SLSQP', 'COBYLA','L-BFGS-B','BFGS']:
            print('Invalid optimization method, try: SLSQP, COBYLA, L-BFGS-B or BFGS')
            exit()
        self.options={}
        
        self.update_options(ftol=ftol, gtol=gtol, rhobeg=rhobeg)

    def update_options(self,ftol,gtol,rhobeg) -> None:
        """Update the optimization options"""

        if self.method in ['SLSQP','L-BFGS-B']:
            self.options['ftol']=ftol
        if self.method in ['L-BFGS-B','BFGS']:
            self.options['gtol']=gtol
        if self.method == 'COBYLA':
            self.options['rhobeg']=rhobeg


class QP_ADAPTVQE(VQE):
    """
    Child class to define the ADAPT VQE.

    Attributes:
        ansatz (ADAPTAnsatz): ADAPT Ansatz object.
        nucleus (Nucleus): Nucleus object.
        parameters (list): List of parameters.
        max_layers (int): Maximum number of layers.
    
    Methods:
        run: Runs the ADAPT VQE algorithm.
        callback: Callback function to store the energy and parameters at each iteration and stop the optimization if the threshold is
    """
    def __init__(self, 
                 ansatz: QP_ADAPTAnsatz,
                 method: str = 'L-BFGS-B',
                 conv_criterion: str = 'Repeated op',
                 test_threshold: float = 1e-2,
                 max_layers: int = 100) -> None:
        
        super().__init__(test_threshold = test_threshold, method = method)
        self.ansatz = ansatz
        self.nucleus = ansatz.nucleus
        self.parameters = []
        self.max_layers = max_layers

        try:
            self.conv_criterion = conv_criterion
        except conv_criterion not in ['Repeated op', 'Gradient','None']:
            print('Invalid minimum criterion. Choose between "Repeated op", "Gradient" and "None"')
            exit()
    
    def run(self) -> tuple:
        """
        Runs the ADAPT VQE algorithm and returns the data of the optimization.

        Returns:
            list: List of the selected operator per layer.
            list: List of energy gradient after optimization per layer.
            list: List of energies per layer.
            list: List of relative errors per layer.
            list: List of function calls per layer.        
        """
 
        print(" --------------------------------------------------------------------------")
        print("                            ADAPT for ", self.nucleus.name)                 
        print(" --------------------------------------------------------------------------\n")

    

        

        E0 = self.ansatz.energy(self.parameters)
        self.energy.append(E0)
        self.rel_error.append(abs((E0 - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))

        print('Initial Energy: ',E0)
        self.ansatz.parameters=[]
        next_operator, next_gradient = self.ansatz.choose_operator()
        gradient_layers = []
        
        energy_layers = [E0]
        rel_error_layers = [self.rel_error[-1]]
    
        
        max_ex = False
        
        while self.ansatz.minimum == False and len(self.ansatz.added_operators)<self.max_layers:
            self.ansatz.added_operators.append(next_operator)
            gradient_layers.append(next_gradient)
            self.parameters.append(0)
            
            try:
                result = minimize(self.ansatz.energy,
                                  self.parameters,
                                  method=self.method,
                                  callback=self.callback,
                                  options=self.options)
                self.parameters = list(result.x)

                self.ansatz.parameters = self.parameters
                
                if len(self.ansatz.added_operators) < self.max_layers:
                    
                    next_operator, next_gradient = self.ansatz.choose_operator()
                     
                    if next_operator == self.ansatz.added_operators[-1]:
                        self.ansatz.minimum = True
                    elif abs(next_gradient) < self.test_threshold:
                        self.ansatz.minimum = True
                    else:
                        energy_layers.append(self.energy[-1])
                        rel_error_layers.append(self.rel_error[-1])

                
            except OptimizationConvergedException:
                pass
            except RuntimeError:
                print('Maximum number of executions reached')
                max_ex=True
            except Exception as e:
                print(e)
                print('FALLO: ', self.parameters)
                
                  
            rel_error = abs((self.energy[-1] - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0])
            
            if rel_error < self.test_threshold or len(self.parameters) == self.max_layers or max_ex==True:
                self.success = True
                self.ansatz.minimum = True
                break
            
            print(f"\n------------ LAYER {len(energy_layers)-1} ------------")
            print('Operator:',self.ansatz.added_operators[-1].A, self.ansatz.added_operators[-1].B,', Gradient:', gradient_layers[-1])
            print('Energy: ',energy_layers[-1])
            print('Rel. Error: ',rel_error_layers[-1])
            print('Theta:', self.parameters[-1])
            
        energy_layers.append(self.energy[-1])
        rel_error_layers.append(self.rel_error[-1])

        print(f"\n------------ LAYER {len(self.parameters)} ------------")
        print('Operator:',self.ansatz.added_operators[-1].A, self.ansatz.added_operators[-1].B,', Gradient:', gradient_layers[-1])
        print('Energy: ',energy_layers[-1])
        print('Rel. Error: ',rel_error_layers[-1])
        print('Theta:', self.parameters[-1])
    
       
        print("\nOperators used for each layer:")
        for i, op in enumerate(self.ansatz.added_operators):
            print(f"Layer {i}: Operator {op.A}{op.B}, Theta = {self.parameters[i]}, Gradient = {gradient_layers[i]}")
            


        print(f'\n Final energy result: {energy_layers[-1]}\t', f'Final relative error is {self.rel_error[-1]}' )

        
        
        if self.conv_criterion == 'None' and self.ansatz.minimum == False:
            self.ansatz.minimum = True
                    
        
        data={'parameters':self.parameters,
            'used_operators':[op for op in self.ansatz.added_operators],
            'operator_pool':self.ansatz.operator_pool,
            'Energy': energy_layers[-1]}
        
        return data
        
    def callback(self, params: list) -> None:
        """
        Callback function to store the energy and parameters at each iteration and stop the optimization if the threshold is reached.
        """
       
        E = self.ansatz.last_energy
        
        self.energy.append(E)
        self.rel_error.append(abs((E - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))
        self.parameters = params
        
        if self.rel_error[-1] < self.test_threshold:
            self.success = True
            self.ansatz.minimum = True
            raise OptimizationConvergedException

def QP_ADAPT_minimization(nucleus: str,
                       ref_state: int = 0,
                       opt_method: str = "L-BFGS-B",
                       threshold: float = 1e-6,
                       max_layers: int = 20,
                       n_qubits: int = 6):

    nuc = QuasiparticleNucleus(nucleus, n_qubits=n_qubits)
    ref_state = np.eye(nuc.d_H)[ref_state]
    
    ansatz = QP_ADAPTAnsatz(nucleus = nuc,
                       ref_state = ref_state)
    
    vqe = QP_ADAPTVQE(ansatz = ansatz,
                   method = opt_method,
                   test_threshold = threshold,
                   max_layers = max_layers)


    data=vqe.run()
    
    
    print('Number of energy measurements:', ansatz.energy_calls)
    return data, nuc

def QP_Quantum_ADAPT_minimization(nucleus: str,
                       ref_state: int = 0,
                       threshold: float = 1e-2,
                       max_layers: int = 20,
                       n_qubits: int = 6, 
                       exact: bool = True,
                       nshots: int = 1000,
                       max_executions: int = 1000):

    nuc = QuasiparticleNucleus(nucleus, n_qubits=n_qubits)
    
    ansatz = QP_QuantumADAPTAnsatz(nucleus = nuc,
                                 ref_state = ref_state,
                                 exact = exact,
                                 nshots = nshots,
                                 max_executions=max_executions)
    
    if exact:
        opt = 'L-BFGS-B'
    else:
        opt = 'COBYLA'
        
    vqe = QP_ADAPTVQE(ansatz = ansatz,
                   method = opt,
                   test_threshold = threshold,
                   max_layers=max_layers)



    data=vqe.run()
    
    print('Number of energy measurements:', ansatz.energy_calls)
   
    return data, nuc

class ADAPT_mixed_VQE(VQE):
    """
    Child class to define the ADAPT VQE.

    Attributes:
        ansatz (ADAPTAnsatz): ADAPT Ansatz object.
        nucleus (Nucleus): Nucleus object.
        parameters (list): List of parameters.
        max_layers (int): Maximum number of layers.
    
    Methods:
        run: Runs the ADAPT VQE algorithm.
        callback: Callback function to store the energy and parameters at each iteration and stop the optimization if the threshold is
    """
    def __init__(self,
                 data: dict, 
                 ansatz: QP_ADAPT_mixed_Ansatz,
                 method: str = 'COBYLA',
                 test_threshold: float = 1e-4,
                 max_layers: int = 100,
                 exact:bool = True) -> None:
        
        super().__init__(test_threshold = test_threshold, method = method)
        self.ansatz = ansatz
        self.nucleus = ansatz.nucleus
        self.parameters = []
    
        
        self.max_layers = max_layers
        self.data = data
        self.exact = exact
        self.operators_used = data['used_operators']
        
        
        
    def run(self) -> tuple:
        """
        Runs the ADAPT VQE algorithm and returns the data of the optimization.

        Returns:
            list: List of the selected operator per layer.
            list: List of energy gradient after optimization per layer.
            list: List of energies per layer.
            list: List of relative errors per layer.
            list: List of function calls per layer.        
        """
 
        print(" --------------------------------------------------------------------------")
        print("                            ADAPT for ", self.nucleus.name)                 
        print(" --------------------------------------------------------------------------\n")


        
        E0 = self.ansatz.energy(self.parameters)
        self.energy.append(E0)
        self.rel_error.append(abs((E0 - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))
        
        
        print('Initial Energy: ',E0)
        next_operator = self.ansatz.choose_operator()
        
        
        energy_layers = [E0]
        rel_error_layers = [self.rel_error[-1]]

        max_ex = False
        
        while self.ansatz.capas<len(self.operators_used) and self.ansatz.capas<self.max_layers:
            
            self.ansatz.capas += 1
            
            self.ansatz.added_operators.append(next_operator)
            

            self.parameters.append(0.0)
            
           
            try:
            
                result = minimize(self.ansatz.energy,
                                    self.parameters,
                                    method=self.method,
                                    callback=self.callback,
                                    options=self.options)
                
                self.parameters = list(result.x)
                
                
                if len(self.parameters)<len(self.data['parameters']): 
                    next_operator = self.ansatz.choose_operator()
                    energy_layers.append(self.energy[-1])
                    rel_error_layers.append(self.rel_error[-1])  
                
            except OptimizationConvergedException:
                pass  
            except RuntimeError:
                print('Maximum number of executions reached')
                max_ex=True 
            except Exception as e:
                print(e)
                print('FALLO: ', self.parameters)
                
               
            rel_error = abs((self.energy[-1] - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0])
            
            if rel_error < self.test_threshold or max_ex==True:
                self.success = True

                self.ansatz.minimum = True
                break
            print(f"\n------------ LAYER {len(energy_layers)-1} ------------")
            print('Operator:',self.ansatz.added_operators[-1].A,self.ansatz.added_operators[-1].B)
            print('Energy: ',energy_layers[-1])
            print('Rel. Error: ',rel_error_layers[-1])
            print('Theta:', self.parameters[-1])

        energy_layers.append(self.energy[-1])
        rel_error_layers.append(self.rel_error[-1])
        print(f"\n------------ LAYER {len(energy_layers)-1} ------------")
        print('Operator:',self.ansatz.added_operators[-1].A,self.ansatz.added_operators[-1].B)
        print('Energy: ',energy_layers[-1])
        print('Rel. Error: ',rel_error_layers[-1])
        print('New operator: ',self.ansatz.added_operators[-1].A, self.ansatz.added_operators[-1].B,'    Theta:', self.parameters[-1])
           
        print("\nOperators used for each layer:")
        
        for i in range(len(self.parameters)):
            print(f"Layer {i+1}: Operator {self.ansatz.added_operators[i].A}{self.ansatz.added_operators[i].B}, Theta = {self.parameters[i]}")

        print(f'\n Final energy result: {energy_layers[-1]}\t', f'Final relative error is {self.rel_error[-1]}' )

        
        data={'parameters':self.parameters,
            'used_operators':[[op.A,op.B] for op in self.ansatz.added_operators],
            'operator_pool':self.ansatz.operator_pool,
            'Energy': energy_layers[-1]}
            
        return data
        
    def callback(self, params: list) -> None:
        """
        Callback function to store the energy and parameters at each iteration and stop the optimization if the threshold is reached.
        """
        
        E = self.ansatz.last_energy
        
        self.energy.append(E)
        self.rel_error.append(abs((E - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))
        self.parameters = params
        
        if self.rel_error[-1] <= self.test_threshold:
            self.success = True
            self.ansatz.minimum = True
            raise OptimizationConvergedException

def QP_ADAPT_mixed_minimization(data: dict,
                            nucleus: QuasiparticleNucleus,
                       ref_state: int = 0,
                       threshold: float = 1e-2,
                       max_layers: int = 20,
                       exact:bool =True,
                       nshots:int = 1000,
                       max_executions:int =100):

    if exact:
        opt = 'L-BFGS-B'
    else:
        opt = 'COBYLA'
        
    ansatz = QP_ADAPT_mixed_Ansatz(data = data,
                                nucleus = nucleus,
                                ref_state = ref_state,
                                exact = exact,
                                nshots = nshots,
                                max_executions=max_executions)
    
    vqe = ADAPT_mixed_VQE(data = data,
                          ansatz = ansatz,
                          method = opt,
                          test_threshold = threshold,
                          max_layers = max_layers,
                          exact = exact)

    data = vqe.run()
 
    
    print('Number of energy measurements:', ansatz.energy_calls)
    return data, nucleus



class QP_Roto_ADAPTVQE(VQE):
    """
    Child class to define the ADAPT VQE.

    Attributes:
        ansatz (ADAPTAnsatz): ADAPT Ansatz object.
        nucleus (Nucleus): Nucleus object.
        parameters (list): List of parameters.
        max_layers (int): Maximum number of layers.
    
    Methods:
        run: Runs the ADAPT VQE algorithm.
        callback: Callback function to store the energy and parameters at each iteration and stop the optimization if the threshold is
    """
    def __init__(self, 
                 ansatz: QP_Roto_ADAPT_Ansatz,
                 method: str = 'L-BFGS-B',
                 conv_criterion: str = 'Repeated op',
                 test_threshold: float = 1e-2,
                 max_layers: int = 100) -> None:
        
        super().__init__(test_threshold = test_threshold, method = method)
        self.ansatz = ansatz
        self.nucleus = ansatz.nucleus
        self.parameters = []
        self.max_layers = max_layers

        try:
            self.conv_criterion = conv_criterion
        except conv_criterion not in ['Repeated op', 'Gradient','None']:
            print('Invalid minimum criterion. Choose between "Repeated op", "Gradient" and "None"')
            exit()
    
    def run(self) -> tuple:
        """
        Runs the ADAPT VQE algorithm and returns the data of the optimization.

        Returns:
            list: List of the selected operator per layer.
            list: List of energy gradient after optimization per layer.
            list: List of energies per layer.
            list: List of relative errors per layer.
            list: List of function calls per layer.        
        """
 
        print(" --------------------------------------------------------------------------")
        print("                            ADAPT for ", self.nucleus.name)                 
        print(" --------------------------------------------------------------------------\n")

    

        

        E0 = self.ansatz.energy(self.parameters)
        self.energy.append(E0)
        rel_error = (abs((E0 - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))

        print('Initial Energy: ',E0)
        self.ansatz.parameters=[]
        next_operator, min_energy, best_parameter = self.ansatz.choose_operator()

        energy_layers = [E0]
        rel_error_layers = [rel_error]
        
        max_ex = False
        
        while self.ansatz.minimum == False and len(self.ansatz.added_operators)<self.max_layers:
            self.ansatz.added_operators.append(next_operator)
            self.parameters.append(best_parameter)
            current_layer_params = self.ansatz.parameters + [best_parameter]           
            
            current_layer_energy = min_energy
            rel_error = (abs((current_layer_energy- self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))
            self.ansatz.set_optimized_state(current_layer_params, current_layer_energy)
            
            try:
                if len(self.parameters)>1:
                    
                    result = minimize(self.ansatz.energy,
                                    self.parameters,
                                    method=self.method,
                                    callback=self.callback,
                                    options=self.options)
                    self.parameters = list(result.x)
                    
                    current_layer_params = list(result.x)
                    current_layer_energy = result.fun
                    energy_layers.append(current_layer_energy)
                    rel_error_layers.append(rel_error)
                    
    
                self.ansatz.parameters = self.parameters
                
                if len(self.ansatz.added_operators) < self.max_layers:
                    
                    next_operator, min_energy, best_parameter = self.ansatz.choose_operator()
                        
                    if next_operator == self.ansatz.added_operators[-1]:
                        self.ansatz.minimum = True
                    elif abs(energy_layers[-1]-min_energy) < self.test_threshold:
                        self.ansatz.minimum = True
                    else:
                        energy_layers.append(current_layer_energy)
                        rel_error_layers.append(rel_error)

                
            except OptimizationConvergedException:
                current_layer_params = self.parameters
                current_layer_energy = self.energy[-1]
                energy_layers.append(current_layer_energy)
                rel_error_layers.append(rel_error)
                 
                pass
            except RuntimeError:
                print('Maximum number of executions reached')
                max_ex=True
            except Exception as e:
                print(e)
                print('FALLO: ', self.parameters)
                
            self.ansatz.set_optimized_state(current_layer_params, current_layer_energy)       
            rel_error = abs((current_layer_energy - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0])
            
        
            if rel_error < self.test_threshold or len(self.parameters) == self.max_layers or max_ex==True:
                self.success = True
                self.ansatz.minimum = True
                break
            
            print(f"\n------------ LAYER {len(self.parameters)} ------------")
            print('Operator:',self.ansatz.added_operators[-1].A,self.ansatz.added_operators[-1].B)
            print('Energy: ',energy_layers[-1])
            print('Rel. Error: ',rel_error)
            print('Theta:', self.parameters[-1])
            
        

        print(f"\n------------ LAYER {len(self.parameters)} ------------")
        print('Operator:',self.ansatz.added_operators[-1].A,self.ansatz.added_operators[-1].B)
        print('Energy: ',current_layer_energy)
        print('Rel. Error: ',rel_error_layers[-1])
        print('Theta:', current_layer_params[-1])
    
       
        print("\nOperators used for each layer:")
        for i, op in enumerate(self.ansatz.added_operators):
            print(f"Layer {i}: Operator {op.A}{op.B}, Theta = {self.parameters[i]}")
            


        print(f'\n Final energy result: {energy_layers[-1]}\t', f'Final relative error is {rel_error}' )

        print(energy_layers)
        
        if self.conv_criterion == 'None' and self.ansatz.minimum == False:
            self.ansatz.minimum = True
                    
        
        data={'parameters':self.parameters,
            'used_operators':[op for op in self.ansatz.added_operators],
            'operator_pool':self.ansatz.operator_pool,
            'Energy': energy_layers[-1]}
        
        return data
        
    def callback(self, params: list) -> None:
        """
        Callback function to store the energy and parameters at each iteration and stop the optimization if the threshold is reached.
        """
       
        E = self.ansatz.last_energy
        
        self.energy.append(E)
        rel_error = abs((E - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0])
        self.parameters = params
        
        if rel_error < self.test_threshold:
            self.success = True
            self.ansatz.minimum = True
            raise OptimizationConvergedException

def QP_Roto_ADAPT_minimization(nucleus: str,
                       ref_state: int = 0,
                       opt_method: str = "COBYLA",
                       threshold: float = 1e-2,
                       max_layers: int = 20,
                       n_qubits: int = 6):

    nuc = QuasiparticleNucleus(nucleus, n_qubits=n_qubits)
    ref_state = np.eye(nuc.d_H)[ref_state]
    
    ansatz = QP_Roto_ADAPT_Ansatz(nucleus = nuc,
                       ref_state = ref_state)
    
    vqe = QP_Roto_ADAPTVQE(ansatz = ansatz,
                   method = opt_method,
                   test_threshold = threshold,
                   max_layers = max_layers)


    data=vqe.run()
    
    
    print('Number of energy measurements:', ansatz.energy_calls)
    return data, nuc

def QP_Quantum_Roto_ADAPT_minimization(nucleus: str,
                       ref_state: int = 0,
                       threshold: float = 1e-2,
                       max_layers: int = 20,
                       n_qubits: int = 6, 
                       exact: bool = True,
                       nshots: int = 1000,
                       max_executions: int = 1000):

    nuc = QuasiparticleNucleus(nucleus, n_qubits=n_qubits)
    
    ansatz = QP_Quantum_Roto_ADAPTAnsatz(nucleus = nuc,
                                 ref_state = ref_state,
                                 exact = exact,
                                 nshots = nshots,
                                 max_executions=max_executions)
    
        
    vqe = QP_Roto_ADAPTVQE(ansatz = ansatz,
                   method = 'COBYLA',
                   test_threshold = threshold,
                   max_layers=max_layers)


    data=vqe.run()
    
    print('Number of energy measurements:', ansatz.energy_calls)
   
    return data, nuc



class QP_Roto_VQE(VQE):
    """
    Child class to define the ADAPT VQE.

    Attributes:
        ansatz (ADAPTAnsatz): ADAPT Ansatz object.
        nucleus (Nucleus): Nucleus object.
        parameters (list): List of parameters.
        max_layers (int): Maximum number of layers.
    
    Methods:
        run: Runs the ADAPT VQE algorithm.
        callback: Callback function to store the energy and parameters at each iteration and stop the optimization if the threshold is
    """
    def __init__(self, 
                 ansatz: QP_Roto_Ansatz,
                 method: str = 'L-BFGS-B',
                 conv_criterion: str = 'Repeated op',
                 test_threshold: float = 1e-2,
                 max_layers: int = 100) -> None:
        
        super().__init__(test_threshold = test_threshold, method = method)
        self.ansatz = ansatz
        self.nucleus = ansatz.nucleus
        self.parameters = []
        self.max_layers = max_layers

        try:
            self.conv_criterion = conv_criterion
        except conv_criterion not in ['Repeated op', 'Gradient','None']:
            print('Invalid minimum criterion. Choose between "Repeated op", "Gradient" and "None"')
            exit()
    
    def run(self) -> tuple:
        """
        Runs the ADAPT VQE algorithm and returns the data of the optimization.

        Returns:
            list: List of the selected operator per layer.
            list: List of energy gradient after optimization per layer.
            list: List of energies per layer.
            list: List of relative errors per layer.
            list: List of function calls per layer.        
        """
 
        print(" --------------------------------------------------------------------------")
        print("                            ADAPT for ", self.nucleus.name)                 
        print(" --------------------------------------------------------------------------\n")

    

        

        E0 = self.ansatz.energy(self.parameters)
        self.energy.append(E0)
        rel_error = (abs((E0 - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))

        self.ansatz.set_optimized_state([], E0)
        print('Initial Energy: ',E0)
        self.ansatz.parameters=[]
        next_operator, min_energy, best_parameter = self.ansatz.choose_operator()
        

        energy_layers = [E0]
        rel_error_layers = [rel_error]
        
        max_ex = False
        
        while self.ansatz.minimum == False and len(self.ansatz.added_operators)<self.max_layers:
            self.ansatz.added_operators.append(next_operator)
            self.parameters.append(best_parameter)
            current_layer_params = self.ansatz.parameters + [best_parameter]           
            
            current_layer_energy = min_energy
            rel_error = (abs((current_layer_energy- self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))
            self.ansatz.set_optimized_state(current_layer_params, current_layer_energy)
            
                    
            
                    
    
            self.ansatz.parameters = self.parameters
                
            if len(self.ansatz.added_operators) < self.max_layers:
                
                next_operator, min_energy, best_parameter = self.ansatz.choose_operator()
                print(min_energy)
                
                
                if next_operator == self.ansatz.added_operators[-1]:
                    self.ansatz.minimum = True
                    energy_layers.append(current_layer_energy)
                    rel_error_layers.append(rel_error)
                elif abs(energy_layers[-1]-min_energy) < self.test_threshold:
                    self.ansatz.minimum = True
                    energy_layers.append(current_layer_energy)
                    rel_error_layers.append(rel_error)
                else:
                    energy_layers.append(current_layer_energy)
                    rel_error_layers.append(rel_error)

                
            self.ansatz.set_optimized_state(current_layer_params, current_layer_energy)       
            rel_error = abs((current_layer_energy - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0])
            
        
            if rel_error < self.test_threshold or len(self.parameters) == self.max_layers or max_ex==True:
                energy_layers.append(current_layer_energy)
                rel_error_layers.append(rel_error)
                self.success = True
                self.ansatz.minimum = True
                break
            
            print(f"\n------------ LAYER {len(self.parameters)} ------------")
            print('Operator:',self.ansatz.added_operators[-1].A,self.ansatz.added_operators[-1].B)
            print('Energy: ',energy_layers[-1])
            print('Rel. Error: ',rel_error)
            print('Theta:', self.parameters[-1])
            
        

        print(f"\n------------ LAYER {len(self.parameters)} ------------")
        print('Operator:',self.ansatz.added_operators[-1].A,self.ansatz.added_operators[-1].B)
        print('Energy: ',current_layer_energy)
        print('Rel. Error: ',rel_error_layers[-1])
        print('Theta:', current_layer_params[-1])
    
       
        print("\nOperators used for each layer:")
        for i, op in enumerate(self.ansatz.added_operators):
            print(f"Layer {i}: Operator {op.A}{op.B}, Theta = {self.parameters[i]}")
            


        print(f'\n Final energy result: {energy_layers[-1]}\t', f'Final relative error is {rel_error}' )

        print(energy_layers)
        
        if self.conv_criterion == 'None' and self.ansatz.minimum == False:
            self.ansatz.minimum = True
                    
        
        data={'parameters':self.parameters,
            'used_operators':[op for op in self.ansatz.added_operators],
            'operator_pool':self.ansatz.operator_pool,
            'Energy': energy_layers[-1]}
        
        return data
        
    def callback(self, params: list) -> None:
        """
        Callback function to store the energy and parameters at each iteration and stop the optimization if the threshold is reached.
        """
       
        E = self.ansatz.last_energy
        
        self.energy.append(E)
        rel_error = abs((E - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0])
        self.parameters = params
        
        if rel_error < self.test_threshold:
            self.success = True
            self.ansatz.minimum = True
            raise OptimizationConvergedException

def QP_Roto_minimization(nucleus: str,
                       ref_state: int = 0,
                       opt_method: str = "COBYLA",
                       threshold: float = 1e-2,
                       max_layers: int = 20,
                       n_qubits: int = 6):

    nuc = QuasiparticleNucleus(nucleus, n_qubits=n_qubits)
    ref_state = np.eye(nuc.d_H)[ref_state]
    
    ansatz = QP_Roto_Ansatz(nucleus = nuc,
                       ref_state = ref_state)
    
    vqe = QP_Roto_VQE(ansatz = ansatz,
                   method = opt_method,
                   test_threshold = threshold,
                   max_layers = max_layers)


    data=vqe.run()
    
    
    print('Number of energy measurements:', ansatz.energy_calls)
    return data, nuc

def QP_Quantum_Roto_minimization(nucleus: str,
                       ref_state: int = 0,
                       threshold: float = 1e-2,
                       max_layers: int = 20,
                       n_qubits: int = 6, 
                       exact: bool = True,
                       nshots: int = 1000,
                       max_executions: int = 1000):

    nuc = QuasiparticleNucleus(nucleus, n_qubits=n_qubits)
    
    ansatz = QP_Quantum_Roto_Ansatz(nucleus = nuc,
                                 ref_state = ref_state,
                                 exact = exact,
                                 nshots = nshots,
                                 max_executions=max_executions)
    
        
    vqe = QP_Roto_VQE(ansatz = ansatz,
                   method = 'COBYLA',
                   test_threshold = threshold,
                   max_layers=max_layers)


    data=vqe.run()
    
    print('Number of energy measurements:', ansatz.energy_calls)
   
    return data, nuc

import numpy as np
from collections import Counter

from openfermion.ops import FermionOperator, QubitOperator
from openfermion.transforms import jordan_wigner
import os

from qibo import gates
from qibo.models.circuit import Circuit
from qibo.result import MeasurementOutcomes

from VQE.Nucleus import Nucleus

import io
import contextlib

class Circuits_Composser():


    
    def __init__(self,
                 nucleus:str='Be6',
                 n_qubits:int = 6,
                 ref_state:int = 0,
                 parameters:list=[],
                 operators_used: list=[],
                 only_ref: bool = False,
                 exact:bool=True,
                 nshots:int=1000):
        
        self.exact = exact
        self.nshots = nshots
        nuc = Nucleus(nucleus, n_qubits=n_qubits)
        
        ham = nuc.Ham_2_body_contributions()
        operator_pool = nuc.operators
        
        two_index=[]
        

        for op in ham:
            if len(set(op.ijkl))==2:
                two_index.append(op)
            


        
        one_body, monoparticular = nuc.Ham_1_body_contributions()
       
        self.monoparticular_energies = monoparticular
        
        self.n_qubits = n_qubits
        self.operator_pool = operator_pool
        self.two_index = two_index
        self.operators_used = operators_used
        self.parameters = parameters
        self.only_ref = only_ref
        self.ref_state = ref_state

        self.name = nuc.name
        self.data_folder = os.path.join(f'nuclei/{self.name}_data')

            
    def Qibo_ref_state_composer(self):
        mb_path = os.path.join(self.data_folder, 'mb_basis_2.dat')

        mb_data = np.loadtxt(mb_path, dtype=str, delimiter=',', skiprows=1)

        ref_state_str = mb_data[self.ref_state]

        ref_state_clean = [int(x) for x in ''.join(ref_state_str).replace('(', ' ').replace(')', ' ').split()]

        ref_state_index = ref_state_clean[1:]

        circuit = Circuit(self.n_qubits)

        for index in ref_state_index:
            circuit.add(gates.X(index))

        for q in range(self.n_qubits):
            if q not in ref_state_index:
                circuit.add(gates.I(q))

        return circuit

    def Fermionic_to_Pauli(self, index):
        """Convierte i (a†_i a†_j a_k a_l - a†_k a†_l a_i a_j) en una cadena de Pauli con n_qubits."""
        i,j,k,l=index
        
        fermion_op = FermionOperator(f"{i}^ {j}^ {k} {l}", 1j)  # i a†_i a†_j a_k a_l
        fermion_op += FermionOperator(f"{k}^ {l}^ {i} {j}", -1j)  # -i a†_k a†_l a_i a_j
        
        pauli_op = jordan_wigner(fermion_op)
        
        
        filtered_pauli_op = QubitOperator()
        for term, coef in pauli_op.terms.items():
            if all(q < self.n_qubits for q, _ in term):  
                filtered_pauli_op += QubitOperator(term, coef)

        return filtered_pauli_op

    def Qibo_staircase_pauli_exponential(self, pauli_op, theta):
        """
        Builds a circuti of e^(i * theta * pauli_op) using the staircase algorithm.

        Parameters:
        - pauli_op: QubitOperator constructed with Pauli_ops
        - n_qubits
        - theta

        Returns:
        - The staircase circuit.
        """

        circuit = Circuit(self.n_qubits)

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
                circuit.add(gates.CNOT(qubit_indices[i], qubit_indices[i+1]))

            # 3. Apply a theta rotation
            last_qubit = qubit_indices[-1]
            circuit.add(gates.RZ(last_qubit, theta=2 * theta * coef.real))

            # 4. Last part staircase structure
            for i in reversed(range(len(qubit_indices) - 1)):
                circuit.add(gates.CNOT(qubit_indices[i], qubit_indices[i+1]))

            # 5. Apply H or Rx gates in the X or Y Pauli exponentials qubits 
            for q, p in term:
                if p == "X":
                    circuit.add(gates.H(q))
                elif p == "Y":
                    circuit.add(gates.RX(q, theta=-np.pi / 2))   

        return circuit

    def Qibo_layer_composer(self):
        """
        Constructs a quantum circuit layer for given fermionic indices using the staircase algorithm for pauli exponentials.
        """

        circuit = Circuit(self.n_qubits)
        for n in range(len(self.operators_used)):
            pauli_op = self.Fermionic_to_Pauli(self.operators_used[n].ijkl)
            circuit += self.Qibo_staircase_pauli_exponential(pauli_op, self.parameters[n])

        return circuit

    def qibo_3_dif_index(self, j, k):
        qubits = list(range(self.n_qubits))
        circuit = Circuit(self.n_qubits)

        circuit.add(gates.CNOT(k, j))
        circuit.add(gates.H(k))
        circuit.add(gates.CNOT(k, j))
        
        return circuit

    def qibo_4_dif_index(self, operator_index):
        qubits = list(range(self.n_qubits))
        circuit = Circuit(self.n_qubits)
        i, j, k, l = operator_index

        circuit.add(gates.CNOT(i, j))
        circuit.add(gates.CNOT(k, i))
        circuit.add(gates.CNOT(l, k))
        circuit.add(gates.H(l))
        circuit.add(gates.CNOT(l, k))
        circuit.add(gates.CNOT(k, i))
        circuit.add(gates.CNOT(i, j))
        
        return circuit

    def Qibo_all_circuits(self):

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
                    circuits_3[jk_sorted] = [circuit, [matrix_element], [repetido], [indexes], jk_sorted]

        for key in circuits_3.keys():
            circuits.append(circuits_3[key])

        return circuits
        
    def Qibo_staircase_pauli_individual_exponential(self, pauli_op):

        circuits=[]
        
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
                circuit_1.add(gates.CNOT(qubit_indices[i], qubit_indices[i+1]))

            # 3. Apply a theta rotation
            last_qubit = qubit_indices[-1]
            circuit_1.add(gates.RZ(last_qubit, theta= np.pi /2))

            # 4. Last part staircase structure
            for i in reversed(range(len(qubit_indices) - 1)):
                circuit_1.add(gates.CNOT(qubit_indices[i], qubit_indices[i+1]))

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
                circuit_2.add(gates.CNOT(qubit_indices[i], qubit_indices[i+1]))

            # 3. Apply a theta rotation
            last_qubit = qubit_indices[-1]
            circuit_2.add(gates.RZ(last_qubit, theta = - np.pi / 2))

            # 4. Last part staircase structure
            for i in reversed(range(len(qubit_indices) - 1)):
                circuit_2.add(gates.CNOT(qubit_indices[i], qubit_indices[i+1]))

            # 5. Apply H or Rx gates in the X or Y Pauli exponentials qubits 
            for q, p in term:
                if p == "X":
                    circuit_2.add(gates.H(q))
                elif p == "Y":
                    circuit_2.add(gates.RX(q, theta=-np.pi / 2))       
            
            circuits.append([circuit_1, circuit_2, coef.real])
            
            
        return circuits
    
    def Qibo_gradient_circuits(self, op_index):       
        
        gradient_circuits=[]
        pauli_op=self.Fermionic_to_Pauli(op_index)
        gradient_apendix=self.Qibo_staircase_pauli_individual_exponential(pauli_op=pauli_op)
        
        for circ in gradient_apendix:
            pairs=[]
            for n in range(2):
                if self.only_ref is False:
                    main_circuit = self.Qibo_ref_state_composer() + self.Qibo_layer_composer() + circ[n]
                else:
                    main_circuit = self.Qibo_ref_state_composer() + circ[n]
                    
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
                            circuits_3[jk_sorted] = [circuit, [matrix_element], [repetido], [indexes], jk_sorted]

                for key in circuits_3.keys():
                    circuits.append(circuits_3[key])
                
                pairs.append(circuits)
            pairs.append(circ[2])  
            gradient_circuits.append(pairs)

        return gradient_circuits

    def obs_P(self, ijkl:list):
        
        obs=[]
        i, j, k, l = ijkl
        
        for m in range(i+1, j):
            if m < k or m > l: 
                obs.append(m)
        
        for n in range(k+1, l):
            if n < i or n > j:  
                obs.append(n)
            
        return obs

    def Qibo_measure_Energy(self, circuits=None):
        
        if circuits == None:
            Qibo_circs = self.Qibo_all_circuits()
        else:
            Qibo_circs=circuits
            
        E_1=0
        E_2_index=0
        main_circ=Qibo_circs[0][0].copy()
        
        if not self.exact:
            
            main_circ.add([gates.M(i) for i in range(main_circ.nqubits)])
            
            result = main_circ.execute(nshots=self.nshots)

            result = MeasurementOutcomes(measurements=result.measurements, backend=result.backend, probabilities=None, samples=result.samples(), nshots=self.nshots)  
        else:
            result=main_circ()
            print(result,'\n')

        
        print('Energy,\t\t\tobservable,\t\tamplitude\n')
        
        for part in self.monoparticular_energies:
            i = int(part)
            eps = self.monoparticular_energies[part]
            
            E_i=eps*result.probabilities([i])[1]
            
            E_1+=E_i
            
            if abs(E_i)>10**(-3):
                print(np.round(E_i,8), '\t\t', i, '\t\t', np.round(eps, 8))
                

        for op in self.two_index:
            
            nu_ij=op.H2b
            ij=set(op.ijkl) 

            E_i=nu_ij*result.probabilities(ij)[3]       
            E_2_index+=E_i
            
            if abs(E_i)>10**(-3):
                print(np.round(E_i,8), '\t\t', op.ijkl, '\t\t', np.round(nu_ij, 8))
    
        
        En_4_index=0
        En_3_index=0



        for i in range(1, len(Qibo_circs)):
            circ=Qibo_circs[i][0].copy()
           
                
            if not self.exact:
                circ.add([gates.M(i) for i in range(circ.nqubits)])
                result = circ.execute(nshots=self.nshots)
                
                res = MeasurementOutcomes(measurements=result.measurements, backend=result.backend, probabilities=None, samples=result.samples(), nshots=self.nshots)   
            else:   
                res=circ()
                
            if len(Qibo_circs[i]) == 3: #the data structure for the 4 diferent index just has 3 parameters
                
                op_amplitude = (Qibo_circs[i][1])
                op_index = Qibo_circs[i][2]     
                
                indices = self.obs_P(op_index)
                
                bitstrings=[]
                for numero in range(2**len(indices)):
                    bitstring = bin(numero)[2:]
                    bitstrings.append(bitstring)
                
                ordenes=indices+op_index
                
                E_ijkl = 0
                
                for bitstring in bitstrings: 
                    n = 0
                    for bit in bitstring:
                        n += int(bit)
                    
                    sign = (-1) ** n  # Alterna entre sumar y restar
                    
                    E_ijkl += sign * op_amplitude * (
                        res.probabilities(ordenes)[int(bitstring+'1100', 2)] - 
                        res.probabilities(ordenes)[int(bitstring+'0011', 2)]
                    )
                    
                    
                En_4_index += E_ijkl 
            
                if abs(E_ijkl)>10**(-3):
                    print(np.round(E_ijkl,8), '\t\t', op_index, '\t\t', np.round(op_amplitude,8))
                    
                    
            elif len(Qibo_circs[i]) == 5:
                
                op_amplitudes = Qibo_circs[i][1]
                reps = Qibo_circs[i][2]
                j,k = Qibo_circs[i][4]
                op_indexs = Qibo_circs[i][3]
                
                for r in range(len(reps)):
                    rep = reps[r]
                    op_index = op_indexs[r]
                    indices = self.obs_P(op_index)

                    op_index = [rep, j, k]
                    op_amplitude=(op_amplitudes[r])
                    
                    bitstrings=[]
                    for numero in range(2**len(indices)):
                        bitstring = bin(numero)[2:]
                        bitstrings.append(bitstring)
                    
                    ordenes = indices + op_index
                    
                    E_njnk = 0
                    
                    for bitstring in bitstrings: 
                        par = 0
                        for bit in bitstring:
                            par += int(bit)
                        
                        sign = (-1) ** par  # Alterna entre sumar y restar
                        
                        E_njnk += sign * op_amplitude * (
                            res.probabilities(ordenes)[int(bitstring+'110', 2)] - 
                            res.probabilities(ordenes)[int(bitstring+'101', 2)]
                        )
                        
                    En_3_index += E_njnk
                    
                    if abs(E_njnk)>10**(-3):
                        print(np.round(E_njnk,8), '\t\t', op_index, '\t\t', np.round(op_amplitude, 8))
                        

        
        print('\n-----------------------------------Final Energies-----------------------------------\n')
        print('Monoparticular energies:                 ', E_1)
        print('Two body energies with 2 diferent index: ', E_2_index)
        print('Two body energies with 3 diferent index: ', En_3_index)
        print('Two body energies with 4 diferent index: ', En_4_index)

        Et=E_1+E_2_index+En_3_index+En_4_index
        
        return Et

    def Qibo_measure_gradient(self, op_index):
        
        gradient=0
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
            gradient=gradient+(en1-en2)*circs[2]

        return gradient

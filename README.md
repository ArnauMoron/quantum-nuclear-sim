# ADAPT-VQE for the nuclear ground state problem

This is the code developed for my Bachelor's and Master's Thesis, which consists in the implementation and simulation of the **ADAPT-VQE** algorithm and its variants to compute the ground state energy of light atomic nuclei.

The algorithm has a classical framework that allows to select the operators and a quantum part that is in charge of building the circuits and measuring the energy. The code includes both **fermionic** and **quasiparticle** encodings, as well as a novel **Roto-ADAPT** variant. Each implementation can be run in three different ways:

* **Classical**: Operator selection and parameter optimization are performed classically.
* **Quantum**: Both operator selection (via quantum gradient calculation) and parameter optimization are performed on a quantum simulator.
* **Mixed**: The operator ordering from a classical run is used to re-optimize the parameters in a quantum simulation environment.

## Fermionic Encoding (`ADAPT_VQE/`)

### `Nucleus.py`

In this file, the `Nucleus` class is implemented. It is in charge of managing the data of a given nucleus. It takes the Hamiltonian, the basis states and the operators from data files. So the code can be easily used to simulate any nuclear shell by changing the number of qubits and the corresponding data files. The class provides both one-body and two-body fermionic excitation operators with their matrix representations and commutators.

### `Ansatze.py`

This file defines the classes for the ansätze:

* `ADAPTAnsatz`: Implements the ADAPT ansatz for the classical version of the algorithm.
* `QuantumADAPTAnsatz`: Implements the ansatz for the **fully quantum version**, where both the energy and the gradients are computed with `qibo`.
* `ADAPT_mixed_Ansatz`: Implements the ansatz for the hybrid or mixed simulation, which uses a predefined operator ordering to optimize parameters on a quantum simulator.

### `Circuit.py`

This file contains the logic for **quantum circuit composition**. The `Circuits_Composser` class is the core of this section, handling:

* The transformation of fermionic operators into Pauli strings (Jordan-Wigner).
* The construction of the exponentials of Pauli operators (the ansatz) using the "staircase" algorithm.
* **Dynamic composition of circuits**: It assembles all the necessary circuits to measure each term of the Hamiltonian (including 2-index, 3-index, and 4-index terms) and the gradients of the operators.

### `Methods.py`

This file implements the main logic of the VQE algorithm. It contains the `ADAPTVQE` and `ADAPT_mixed_VQE` classes, which manage the optimization loop and interface with the corresponding ansatz. The code handles the entire quantum workflow, including calling the circuit composer for energy and gradient measurements. High-level functions like `ADAPT_minimization()`, `Quantum_ADAPT_minimization()`, and `ADAPT_mixed_minimization()` provide easy access to different simulation modes.

## Quasiparticle Encoding (`QP_ADAPT_VQE/`)

### `Nucleus.py`

This file extends the fermionic framework with the `QuasiparticleNucleus` class, which:

* Maps fermionic pairs to quasiparticle modes based on angular momentum quantum numbers.
* Analytically derives Hamiltonian coefficients in the quasiparticle basis.
* Provides one-body, two-body diagonal, and hopping operators.
* Significantly reduces the operator pool size compared to fermionic encoding.

### `Ansatze.py`

This file defines the quasiparticle ansatz classes:

* `QP_ADAPTAnsatz`: Classical ADAPT with quasiparticle operators.
* `QP_QuantumADAPTAnsatz`: Quantum gradient-based operator selection.
* `QP_ADAPT_mixed_Ansatz`: Hybrid simulation with predefined operators.
* `QP_Roto_ADAPT_Ansatz`: **Novel variant** that uses analytical Fourier analysis to find optimal parameters, reducing circuit executions.
* `QP_Quantum_Roto_ADAPTAnsatz`: Quantum circuit version of Roto-ADAPT.
* `QP_Roto_Ansatz`: Fixed operator pool with Roto optimization.
* `QP_Quantum_Roto_Ansatz`: Quantum version with fixed operator pool.

### `Circuit.py`

This file implements quantum circuits for quasiparticle encoding via the `QP_Circuits_Composser` class:

* Uses GIVENS rotations for quasiparticle hopping operators.
* Measures diagonal and off-diagonal Hamiltonian terms efficiently.
* Implements `Qibo_find_min()` for analytical parameter optimization: fits the energy landscape to E(θ) = c₀ + c₁cos(θ) + s₁sin(θ) + c₂cos(2θ) + s₂sin(2θ) using only 5 measurements, then finds the minimum analytically.
* Provides gradient calculation for operator selection.

### `Methods.py`

This file implements VQE optimization for quasiparticle encoding:

* `QP_ADAPTVQE`: Standard ADAPT optimization loop.
* `QP_Roto_ADAPTVQE`: **Roto-ADAPT variant** that analytically determines optimal parameters for each operator, significantly reducing the number of circuit executions.
* `QP_Roto_VQE`: Fixed pool optimization with the Roto method.
* High-level functions: `QP_ADAPT_minimization()`, `QP_Quantum_ADAPT_minimization()`, `QP_Roto_ADAPT_minimization()`, `QP_Quantum_Roto_ADAPT_minimization()`, etc.

## `ADAPT-VQE simulations.ipynb`

This is a Jupyter Notebook that shows the whole workflow of the project. It has examples on how to run the classical, full quantum and mixed simulations for both fermionic and quasiparticle encodings, as well as the analysis of the results. It includes:

* Comparison between standard ADAPT and Roto-ADAPT variants.
* Shot noise analysis with varying numbers of measurements (100 to 10,000 shots).
* Energy measurement decomposition (diagonal vs. hopping contributions).
* Convergence tests and parameter statistics across multiple runs.

## `requirements.txt`

This file contains a list of all the Python dependencies needed to run the code, including `numpy`, `scipy`, `qibo`, `qibojit`, and `openfermion`.

## Acknowledgements

The foundational code for the classical ADAPT-VQE algorithm was developed by **Miquel Carrasco**. His original work can be found in his repository:
[miquel-carrasco/UCC_vs_ADAPT_p_shell](https://github.com/miquel-carrasco/UCC_vs_ADAPT_p_shell). Also, the Hamiltonian matrix elements and many-body basis files have been created with **Antonio Márquez Romero**'s code.

The adaptation of this classical framework and the **entire quantum implementation**—including circuit building with `qibo`, energy measurement protocols, the hybrid simulation logic, the **quasiparticle encoding framework**, and the **Roto-ADAPT optimization algorithm**—were developed as part of my Bachelor's Thesis.

## Contact

For any questions, suggestions, or collaborations regarding this project, feel free to contact me at:

* **Arnau Morón**: [arnau.moron@gmail.com](mailto:arnau.moron@gmail.com)
# ADAPT-VQE for the nuclear ground state problem

This is the code developed for my Bachelor's and Master's Thesis, which consists in the implementation and simulation of the **ADAPT-VQE** algorithm and its variants, plus a **digital quantum annealing** protocol, to compute the ground state energy of light atomic nuclei.

The algorithm has a classical framework that allows to select the operators and a quantum part that is in charge of building the circuits and measuring the energy. The code includes both **fermionic** and **HCB** (hard-CORE boson) encodings, as well as a novel **GGF-ADAPT** variant. Each ADAPT-VQE implementation can be run in three different ways:

* **Classical**: Operator selection and parameter optimization are performed classically.
* **Quantum**: Both operator selection (via quantum gradient calculation) and parameter optimization are performed on a quantum simulator.
* **Mixed**: The operator ordering from a classical run is used to re-optimize the parameters in a quantum simulation environment.

## Repository structure

ADAPT-VQE and annealing are two separate algorithms built on the same nuclear-shell-model data and the same low-level circuit-building infrastructure. That shared infrastructure, and each algorithm, live in their own top-level package, so there is a clear, physical boundary between "code every algorithm depends on" and "code specific to one algorithm":

```
CORE/                   # shared: nothing ADAPT-VQE- or annealing-specific
  Nucleus.py
  CircuitBuilder.py

adapt_vqe/               # ADAPT-VQE only
  Circuit.py
  Ansatze.py
  Methods.py

annealing/                # annealing only
  Annealing.py
  legacy/
    annealing.ipynb        # exploratory notebook the module was worked out from

nuclei/                    # shared data files (both packages read from here)
```

Both `adapt_vqe` and `annealing` depend on `CORE`; neither depends on the other. Every class/function taking an `encoding` argument accepts `'fermionic'` (default) or `'HCB'`, instead of a separate `QP_`-prefixed duplicate: `Nucleus`, `CircuitBuilder`, `Circuits_Composer`, `Ansatz`/`ADAPTAnsatz`/`QuantumADAPTAnsatz`/`ADAPT_mixed_Ansatz`, `VQE`/`ADAPTVQE`/`ADAPT_mixed_VQE`, the GGF-ADAPT family (`GGF_ADAPT_Ansatz`/`Quantum_GGF_ADAPTAnsatz`/`GGF_Ansatz`/`Quantum_GGF_Ansatz`/`GGF_ADAPTVQE`/`GGF_VQE`), every top-level `*_minimization()` function, `AnnealingProtocol`, and `Annealing_simulation`.

## `CORE/Nucleus.py`

* `Nucleus(nuc_name, shell='p', encoding='fermionic')`: Manages the data of a given nucleus. It takes the Hamiltonian, the basis states and the operators from data files, so the code can be easily used to simulate any nuclear shell by changing the number of qubits and the corresponding data files.
  * With `encoding='fermionic'` (default), it provides one-body and two-body fermionic excitation operators with their matrix representations and commutators. The commutator of each operator with the Hamiltonian is precomputed here (via a sparse intermediate, since each operator matrix is extremely sparse) so it can be reused directly for gradient evaluations instead of recomputing it every VQE iteration.
  * With `encoding='HCB'`, it builds an internal fermionic `Nucleus` (`self.nucleus`) and, from it, maps fermionic pairs to hard-CORE boson modes based on angular momentum quantum numbers, analytically derives Hamiltonian coefficients in the HCB basis, and provides one-body, two-body diagonal, and hopping operators (`HCB_OneBodyOperator`, `HCB_TwoBodyDiagonalOperator`, `HCB_HoppingOperator`) — significantly reducing the operator pool size compared to the fermionic encoding.
  * Many-body basis states are indexed through a dictionary built once at construction time, so building the operator pool and projecting the Hamiltonian only ever needs O(1) state lookups rather than repeated linear scans of the basis.

## `CORE/CircuitBuilder.py`

* `CircuitBuilder(nucleus, ref_state=0, encoding='fermionic')`: the circuit-building infrastructure genuinely shared by ADAPT-VQE and annealing — nothing here is specific to either. Determines `n_qubits`/qubit re-basing for the given encoding, and provides:
  * `Qibo_ref_state_composer()`: builds the reference-state circuit (both encodings).
  * `Qibo_staircase_pauli_exponential()`: builds `e^(i·θ·pauli_op)` via the CNOT "staircase" algorithm, for an arbitrary multi-qubit Pauli string.
  * `Observable_index_to_Pauli()` / `OneBody_index_to_Pauli()`: Jordan-Wigner conversions of the Hermitian two-body and one-body Hamiltonian pieces (fermionic).
  * `_local()` / `_local_list()`: raw single-particle index → local 0-indexed circuit qubit position.

  `adapt_vqe.Circuit.Circuits_Composer` and `annealing.Annealing.AnnealingProtocol` both subclass this directly — ADAPT-VQE's much heavier operator-pool/gradient/GGF-measurement machinery and annealing's Hamiltonian-interpolation/Trotter-circuit machinery each build on the same base without either depending on the other.

## `adapt_vqe/Ansatze.py`

All ansatz classes take `encoding='fermionic'` (default) or `encoding='HCB'`:

* `ADAPTAnsatz`: Implements the ADAPT ansatz for the classical version of the algorithm. Operator gradients are evaluated as `⟨ψ|[H, op]|ψ⟩` using the operator's precomputed commutator, for both encodings.
* `QuantumADAPTAnsatz`: Implements the ansatz for the **fully quantum version**, where both the energy and the gradients are computed with `qibo`.
* `ADAPT_mixed_Ansatz`: Implements the ansatz for the hybrid or mixed simulation, which uses a predefined operator ordering to optimize parameters on a quantum simulator.
* `GGF_ADAPT_Ansatz`: **Novel variant** that uses analytical Fourier analysis to find optimal parameters, reducing circuit executions.
* `Quantum_GGF_ADAPTAnsatz`: Quantum circuit version of GGF-ADAPT. For the HCB encoding the analytic fit is applied directly on the GIVENS-rotation landscape; for the fermionic encoding it's applied on the landscape produced by the staircase Pauli-exponential circuit for the candidate operator.
* `GGF_Ansatz`: Fixed operator pool with GGF optimization.
* `Quantum_GGF_Ansatz`: Quantum version with fixed operator pool.

## `adapt_vqe/Circuit.py`

* `Circuits_Composer(..., encoding='fermionic')`: subclasses `CORE.CircuitBuilder`, adding the ADAPT-VQE-specific circuit-measurement machinery (operator pools, ansatz layers, gradient/GGF measurement) on top of the shared reference-state/staircase/Jordan-Wigner infrastructure.
  * With `encoding='fermionic'`, it transforms fermionic operators into Pauli strings (Jordan-Wigner), builds the exponentials of Pauli operators (the ansatz) using the "staircase" algorithm, and dynamically composes the circuits needed to measure each term of the Hamiltonian (including 2-index, 3-index, and 4-index terms) and the gradients of the operators. For exact (noiseless) energy, gradient, and GGF parameter-offset evaluations, the shared ansatz (or ansatz+offset/shift) circuit is simulated once and its statevector is cached and reused as the `initial_state` for each Hamiltonian term's much smaller measurement-basis circuit, instead of re-simulating the whole thing from scratch per term.
  * With `encoding='HCB'`, it uses GIVENS rotations for the hopping operators and measures diagonal and off-diagonal Hamiltonian terms efficiently. For reference states in the "complex" regime (roughly, anything other than a nearly-empty or nearly-full occupation — see `Circuits_Composer.complex`), exact evaluations use the same cached-statevector strategy across every hopping-pair term instead of resimulating the ansatz per pair, and the non-diagonal XX+YY hopping terms are measured with one of two strategies selected by `hop_measurement` (default `'givens'`) — see **"Measuring XX+YY: Givens rotation vs. Hadamard"** below.
  * For both encodings it implements `Qibo_find_min()` for analytical parameter optimization (fits the energy landscape to E(θ) = c₀ + c₁cos(θ) + s₁sin(θ) + c₂cos(2θ) + s₂sin(2θ) using only 5 measurements, then finds the minimum analytically) — the 5 circuits it measures are built via GIVENS rotations for HCB or via the staircase protocol for the fermionic encoding.
  * See **"Statevector caching vs. hardware-realistic execution"** below for how to disable this caching.

### Measuring XX+YY: Givens rotation vs. Hadamard

For HCB reference states in the "complex" regime, each non-diagonal Hamiltonian term `Vᵢⱼ(XᵢXⱼ+YᵢYⱼ)` is measured with the project's Z-basis reconstruction technique: the term's **magnitude** is always recovered from the *unrotated* Z-basis coherence, `2√(P₀₁P₁₀)` (exact for these real-amplitude states), and only its **sign** needs an extra rotated measurement. `Circuits_Composer(..., hop_measurement=...)` selects which rotation is used for that sign-determining measurement, both described in the project's reference paper (*Bridging Quantum Computing and Nuclear Structure...*, Methods, "Measurement of energy expectation values"):

* `'givens'` (**default**): the paper's "Basis rotation" (BasisRot) strategy. A two-qubit Givens rotation `GIVENS(A, B, -π/4)` is applied to each hopping pair before measuring, which diagonalizes `XᵢXⱼ+YᵢYⱼ` into `Zᵢ-Zⱼ` (paper Eq. 17); the sign of `P₀₁-P₁₀` on the rotated circuit gives the term's sign. Unlike Hadamard, this rotation conserves particle number on the pair.
* `'hadamard'`: the original strategy, matching the paper's "Hadamard" method. Applies Hadamard gates to rotate the pair into the X basis and reads the sign off `⟨XᵢXⱼ⟩` (assuming `⟨XᵢXⱼ⟩=⟨YᵢYⱼ⟩`).

Both strategies compute the same physical quantity — cross-checked to agree to float precision in exact mode — but the Givens/BasisRot strategy needs no separate sign/magnitude reconstruction and, per the paper, is more robust to shot noise. `hop_measurement` is forwarded from `QuantumADAPTAnsatz`, `ADAPT_mixed_Ansatz`, `Quantum_GGF_ADAPTAnsatz`, and `Quantum_GGF_Ansatz` down to their internal `Circuits_Composer`, same as `statevector_caching` below. It only affects the HCB "complex" path — the HCB "simple" path and the fermionic encoding ignore it.

### Statevector caching vs. hardware-realistic execution

By default (`statevector_caching=True`), every exact (`exact=True`) energy/gradient/GGF evaluation described above simulates the ansatz (or ansatz+offset/shift) circuit once and reuses its statevector as `initial_state=` for the smaller circuits that follow, instead of re-simulating the full gate sequence from `|0...0⟩` for every Hamiltonian term or hopping pair. This is a **simulator-only** technique: it is what makes the exact-simulation modes fast, but real quantum hardware (and any hardware-faithful simulation, e.g. one that transpiles to a real backend or applies a realistic noise model) cannot be handed an arbitrary mid-circuit state to resume from — every circuit execution has to start from `|0...0⟩` and run its own complete gate sequence.

Set `statevector_caching=False` to bypass this and always build/execute the full, from-`|0...0⟩` circuit for every measurement — i.e. exactly the sequence of circuits that would need to run on real hardware, at the cost of the corresponding speedup. This is available at every layer of the API:

* `Circuits_Composer(..., statevector_caching=True)`
* `QuantumADAPTAnsatz`, `ADAPT_mixed_Ansatz`, `Quantum_GGF_ADAPTAnsatz`, `Quantum_GGF_Ansatz` (all take `statevector_caching=True`, forwarded to their internal `Circuits_Composer`)
* The corresponding top-level functions in `Methods.py`: `Quantum_ADAPT_minimization()`, `ADAPT_mixed_minimization()`, `Quantum_GGF_ADAPT_minimization()`, `Quantum_GGF_minimization()`

Note this flag only changes *how* an exact evaluation is computed, never *what* it computes — both settings return identical energies/gradients/parameters. It also has no effect on: shot-noise evaluations (`exact=False`), which already sample from freshly, fully executed circuits regardless of this flag; and the HCB "simple" reference-state case (`Circuits_Composer.complex == False`), which was already built from a single shared ansatz circuit with no per-term caching to begin with. In both of those cases the code is already "hardware-realistic" in this specific sense. This flag is specific to `adapt_vqe`'s `Circuits_Composer` — it has no counterpart in `annealing`, whose circuits are always built from `|0...0⟩` gate-by-gate regardless (see `annealing/Annealing.py` below).

## `adapt_vqe/Methods.py`

All optimization classes/functions take `encoding='fermionic'` (default) or `encoding='HCB'`: `ADAPTVQE` and `ADAPT_mixed_VQE` manage the optimization loop and interface with the corresponding ansatz, calling the circuit composer for energy and gradient measurements. `GGF_ADAPTVQE` is the **GGF-ADAPT variant** that analytically determines optimal parameters for each operator (significantly reducing the number of circuit executions), and `GGF_VQE` performs fixed-pool optimization with the GGF method. High-level functions `ADAPT_minimization()`, `Quantum_ADAPT_minimization()`, `ADAPT_mixed_minimization()`, `GGF_ADAPT_minimization()`, `Quantum_GGF_ADAPT_minimization()`, `GGF_minimization()`, and `Quantum_GGF_minimization()` provide easy access to the different simulation modes for both encodings.

The optimizer is chosen from the noise level, not left as a free default: exact/noiseless evaluations use L-BFGS-B, shot-noise evaluations use COBYLA (with scipy's own default `ftol`/`gtol`, not custom-tuned ones).

## `annealing/Annealing.py`

Implements a time-Trotterized **digital quantum annealing** protocol, for either encoding, that interpolates from an easily-preparable driver Hamiltonian `H_D` to the real nuclear Hamiltonian `H_T` and Trotter-simulates `H(s) = (1-s) H_D + s H_T` on a qubit circuit. `AnnealingProtocol` subclasses `CORE.CircuitBuilder` directly for the reference-state circuit, the staircase Pauli-exponential builder, and the Jordan-Wigner conversions it needs — it has no dependency on `adapt_vqe` (no operator pools, no ansatz layers, no gradient/GGF machinery).

* `AnnealingProtocol(nucleus, ref_state=0, encoding='fermionic')`: builds `H_D` (a sum of single-orbital/mode number operators, scaled so its unique ground state in the fixed-particle-number sector is exactly `ref_state`) and `H_T` (the real Hamiltonian, `nucleus.H` in the small many-body basis and as a Pauli-string sum for the circuit), both as small matrices and as `QubitOperator`s.
  * `exact_evolution()`: classical small-matrix reference evolution (exact per-step unitary via `scipy.linalg.expm`), plus `H(s)`'s instantaneous eigenvalues at each step.
  * `build_circuit()`: the direct-gate Trotterized circuit (arbitrary qubit-pair connectivity) - fermionic terms via the staircase Pauli-exponential technique, HCB terms via native `RZ`/`CU1`/`RXXYY` gates.
  * `build_swap_network_circuit()` (HCB only): a nearest-neighbor **swap-network** circuit - only ever applies 2-qubit gates between adjacent physical qubits, sweeping a logical-mode/physical-qubit map back and forth with `SWAP` gates so every pair of modes becomes adjacent at some point, trading circuit depth for hardware-realistic connectivity.
  * `embed_subspace_vector()`: maps any vector expressed in the nucleus's small many-body basis (e.g. its true ground state, `nucleus.eig_vec[:, 0]`) into the full `2**n_qubits` computational-basis statevector, for fidelity comparisons.
  * `run()`: builds and executes the requested circuit variant, returning its final state, energy, and fidelity to the exact ground state.
* `Annealing_simulation(nucleus, ...)`: top-level convenience function mirroring `Methods.py`'s `*_minimization()` pattern.

`annealing/legacy/annealing.ipynb` is the exploratory notebook this module's algorithm was originally worked out in, kept for reference; it targets an older, now-gone package and is superseded by `Annealing.py` and `Annealing simulations.ipynb`.

## `ADAPT-VQE simulations.ipynb`

This is a Jupyter Notebook that shows the whole workflow of the ADAPT-VQE project (`adapt_vqe`). It has examples on how to run the classical, full quantum and mixed simulations for both the fermionic and HCB encodings, as well as the analysis of the results. It includes:

* Comparison between standard ADAPT and GGF-ADAPT variants.
* Shot noise analysis with varying numbers of measurements (100 to 10,000 shots).
* Energy measurement decomposition (diagonal vs. hopping contributions).
* Convergence tests and parameter statistics across multiple runs.

## `Annealing simulations.ipynb`

Showcases the `annealing.Annealing` module: consistency checks (the circuit's Pauli-sum Hamiltonian exactly reproduces `nucleus.H`; the driver Hamiltonian's ground state is exactly the reference state), the fermionic and HCB direct-gate annealing protocols (instantaneous eigenvalues, exact vs. circuit energy, fidelity), and the HCB nearest-neighbor swap-network circuit compared against the direct-gate one (fidelity and infidelity-vs-steps convergence).

## `requirements.txt`

This file contains a list of all the Python dependencies needed to run the code, including `numpy`, `scipy`, `qibo`, `qibojit`, and `openfermion`.

## Acknowledgements

The foundational code for the classical ADAPT-VQE algorithm was developed by **Miquel Carrasco**. His original work can be found in his repository:
[miquel-carrasco/UCC_vs_ADAPT_p_shell](https://github.com/miquel-carrasco/UCC_vs_ADAPT_p_shell). Also, the Hamiltonian matrix elements and many-body basis files have been created with **Antonio Márquez Romero**'s code.

The adaptation of this classical framework and the **entire quantum implementation**—including circuit building with `qibo`, energy measurement protocols, the hybrid simulation logic, the **HCB (hard-CORE boson) encoding framework**, and the **GGF-ADAPT optimization algorithm**—were developed as part of my Bachelor's Thesis.

## Contact

For any questions, suggestions, or collaborations regarding this project, feel free to contact me at:

* **Arnau Morón**: [arnau.moron@gmail.com](mailto:arnau.moron@gmail.com)

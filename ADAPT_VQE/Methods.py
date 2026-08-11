from scipy.optimize import minimize
import numpy as np

from CORE.Nucleus import Nucleus
from ADAPT_VQE.Ansatze import (
    ADAPTAnsatz,
    QuantumADAPTAnsatz,
    ADAPT_mixed_Ansatz,
    GGF_ADAPT_Ansatz,
    Quantum_GGF_ADAPTAnsatz,
    GGF_Ansatz,
    Quantum_GGF_Ansatz,
)


class OptimizationConvergedException(Exception):
    pass


class VQE:
    """
    Parent class to define the Variational Quantum Eigensolvers (VQEs), for either
    the fermionic or the HCB (hard-core boson) encoding.

    Attributes:
        method (str): Optimization method.
        test_threshold (float): Threshold to stop the optimization.
        energy (list): List of energies.
        rel_error (list): List of relative errors.
        success (bool): If True, the optimization was successful.
        tot_operations (list): List of total operations.
        options (dict): Optimization options.
        encoding (str): Encoding used, either 'fermionic' or 'HCB'.

    Methods:
        update_options: Update the optimization options.
    """

    def __init__(
        self,
        test_threshold: float = 1e-4,
        method: str = "L-BFGS-B",
        rhobeg: float = 0.2,
        encoding: str = "fermionic",
    ) -> None:
        """
        Initialization of the VQE object.

        Args:
            test_threshold (float): Threshold to stop the optimization.
            method (str): Optimization method.
            rhobeg (float): Tolerance for the constraints.
            encoding (str): Encoding used, either 'fermionic' or 'HCB'.
        """
        self.encoding = encoding
        self.method = method
        self.test_threshold = test_threshold

        self.energy = []
        self.rel_error = []
        self.success = False
        self.tot_operations = [0]
        try:
            self.method = method
        except method not in ["SLSQP", "COBYLA", "L-BFGS-B", "BFGS"]:
            print("Invalid optimization method, try: SLSQP, COBYLA, L-BFGS-B or BFGS")
            exit()
        self.options = {}

        self.update_options(rhobeg=rhobeg)

    def update_options(self, rhobeg) -> None:
        """Update the optimization options. ftol/gtol are left at their scipy defaults."""

        if self.method == "COBYLA":
            self.options["rhobeg"] = rhobeg


class ADAPTVQE(VQE):
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

    def __init__(
        self,
        ansatz: ADAPTAnsatz,
        method: str = "L-BFGS-B",
        conv_criterion: str = "Repeated op",
        test_threshold: float = 1e-2,
        max_layers: int = 100,
    ) -> None:

        self.encoding = ansatz.encoding
        super().__init__(
            test_threshold=test_threshold, method=method, encoding=self.encoding
        )
        self.ansatz = ansatz
        self.nucleus = ansatz.nucleus
        self.parameters = []
        self.max_layers = max_layers

        self.ofset = 0
        if self.encoding == "HCB":
            if self.nucleus.nucleus.qubits[0] != 0:
                if self.nucleus.shell == "p":
                    self.ofset = 0
                elif self.nucleus.shell == "sd":
                    self.ofset = 0

        try:
            self.conv_criterion = conv_criterion
        except conv_criterion not in ["Repeated op", "Gradient", "None"]:
            print(
                'Invalid minimum criterion. Choose between "Repeated op", "Gradient" and "None"'
            )
            exit()

    def _op_label_args(self, op) -> tuple:
        """Returns the operator's identifying indices, as positional print args."""
        if self.encoding == "fermionic":
            return (op.ijkl,)
        return (op.A + self.ofset, op.B + self.ofset)

    def _op_label_str(self, op) -> str:
        """Returns the operator's identifying indices, formatted for an f-string."""
        if self.encoding == "fermionic":
            return f"{op.ijkl}"
        return f"{op.A+self.ofset},{op.B+self.ofset}"

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

        print(
            " --------------------------------------------------------------------------"
        )
        print("                            ADAPT for ", self.nucleus.name)
        print(
            " --------------------------------------------------------------------------\n"
        )

        E0 = self.ansatz.energy(self.parameters)
        self.energy.append(E0)
        self.rel_error.append(
            abs((E0 - self.ansatz.nucleus.eig_val[0]) / self.ansatz.nucleus.eig_val[0])
        )

        print("Initial Energy: ", E0)
        self.ansatz.parameters = []
        next_operator, next_gradient = self.ansatz.choose_operator()

        gradient_layers = []

        energy_layers = [E0]
        rel_error_layers = [self.rel_error[-1]]

        max_ex = False

        while (
            self.ansatz.minimum == False
            and len(self.ansatz.added_operators) < self.max_layers
        ):
            self.ansatz.added_operators.append(next_operator)
            gradient_layers.append(next_gradient)
            self.parameters.append(0.0)

            try:
                result = minimize(
                    self.ansatz.energy,
                    self.parameters,
                    method=self.method,
                    callback=self.callback,
                    options=self.options,
                )
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
                print("Maximum number of executions reached")
                max_ex = True
            except Exception as e:
                print(e)
                print("FALLO: ", self.parameters)

            rel_error = abs(
                (self.energy[-1] - self.ansatz.nucleus.eig_val[0])
                / self.ansatz.nucleus.eig_val[0]
            )

            if (
                rel_error < self.test_threshold
                or len(self.parameters) == self.max_layers
                or max_ex == True
            ):
                self.success = True
                self.ansatz.minimum = True
                break

            print(f"\n------------ LAYER {len(energy_layers)-1} ------------")
            print(
                "Operator:",
                *self._op_label_args(self.ansatz.added_operators[-1]),
                ", Gradient:",
                gradient_layers[-1],
            )
            print("Energy: ", energy_layers[-1])
            print("Rel. Error: ", rel_error_layers[-1])
            print("Theta:", self.parameters[-1])

        energy_layers.append(self.energy[-1])
        rel_error_layers.append(self.rel_error[-1])

        print(f"\n------------ LAYER {len(self.parameters)} ------------")
        print(
            "Operator:",
            *self._op_label_args(self.ansatz.added_operators[-1]),
            ", Gradient:",
            gradient_layers[-1],
        )
        print("Energy: ", energy_layers[-1])
        print("Rel. Error: ", rel_error_layers[-1])
        print("Theta:", self.parameters[-1])

        print("\nOperators used for each layer:")
        for i, op in enumerate(self.ansatz.added_operators):
            print(
                f"Layer {i}: Operator {self._op_label_str(op)}, Theta = {self.parameters[i]}, Gradient = {gradient_layers[i]}"
            )

        print(
            f"\n Final energy result: {energy_layers[-1]}\t",
            f"Final relative error is {self.rel_error[-1]}",
        )

        if self.conv_criterion == "None" and self.ansatz.minimum == False:
            self.ansatz.minimum = True

        data = {
            "parameters": self.parameters,
            "used_operators": [op for op in self.ansatz.added_operators],
            "operator_pool": self.ansatz.operator_pool,
            "Energy": energy_layers[-1],
        }

        return data

    def callback(self, params: list) -> None:
        """
        Callback function to store the energy and parameters at each iteration and stop the optimization if the threshold is reached.
        """

        E = self.ansatz.last_energy

        self.energy.append(E)
        self.rel_error.append(
            abs((E - self.ansatz.nucleus.eig_val[0]) / self.ansatz.nucleus.eig_val[0])
        )
        self.parameters = params

        if self.rel_error[-1] < self.test_threshold:
            self.success = True
            self.ansatz.minimum = True
            raise OptimizationConvergedException


def ADAPT_minimization(
    nucleus: str,
    ref_state: int = 0,
    opt_method: str = "L-BFGS-B",
    threshold: float = 1e-6,
    max_layers: int = 20,
    shell: str = "p",
    encoding: str = "fermionic",
):

    ref_state_dict = {"ref_state": ref_state}
    nuc = Nucleus(nucleus, shell=shell, encoding=encoding)
    ref_state = np.eye(nuc.d_H)[ref_state]

    ansatz = ADAPTAnsatz(nucleus=nuc, ref_state=ref_state, encoding=encoding)

    vqe = ADAPTVQE(
        ansatz=ansatz,
        method=opt_method,
        test_threshold=threshold,
        max_layers=max_layers,
    )

    data = vqe.run()

    if encoding == "fermionic":
        ham = nuc.Ham_2_body_contributions()

        two_index = []
        ham_pool = []

        for op in ham:
            if len(set(op.ijkl)) == 2:
                two_index.append(op)
            else:
                ham_pool.append(op)

        one_body, monoparticular = nuc.Ham_1_body_contributions()
        diag_two_body_ops = {"two_index": two_index}
        ham_pool_ops = {"ham_pool": ham_pool}
        one_body = {"one_body": one_body}
        monoparticular = {"monoparticular": monoparticular}
        name = {"name": nucleus}

        data.update(monoparticular)

        data.update(diag_two_body_ops)

        data.update(ham_pool_ops)

        data.update(one_body)

        data.update(name)

        data.update(ref_state_dict)

    print("Number of energy measurements:", ansatz.energy_calls)
    return data, nuc


def Quantum_ADAPT_minimization(
    nucleus: str,
    ref_state: int = 0,
    threshold: float = 1e-2,
    max_layers: int = 20,
    shell: str = "p",
    exact: bool = True,
    nshots: int = 1000,
    max_executions: int = 1000,
    encoding: str = "fermionic",
    statevector_caching: bool = True,
):

    ref_state_dict = {"ref_state": ref_state}
    nuc = Nucleus(nucleus, shell=shell, encoding=encoding)

    ansatz = QuantumADAPTAnsatz(
        nucleus=nuc,
        ref_state=ref_state,
        exact=exact,
        nshots=nshots,
        max_executions=max_executions,
        encoding=encoding,
        statevector_caching=statevector_caching,
    )

    if exact:
        opt = "L-BFGS-B"
    else:
        opt = "COBYLA"

    vqe = ADAPTVQE(
        ansatz=ansatz, method=opt, test_threshold=threshold, max_layers=max_layers
    )

    data = vqe.run()

    if encoding == "fermionic":
        ham = nuc.Ham_2_body_contributions()

        two_index = []
        ham_pool = []

        for op in ham:
            if len(set(op.ijkl)) == 2:
                two_index.append(op)
            else:
                ham_pool.append(op)

        one_body, monoparticular = nuc.Ham_1_body_contributions()
        diag_two_body_ops = {"two_index": two_index}
        ham_pool_ops = {"ham_pool": ham_pool}
        one_body = {"one_body": one_body}
        monoparticular = {"monoparticular": monoparticular}
        name = {"name": nucleus}

        data.update(monoparticular)

        data.update(diag_two_body_ops)

        data.update(ham_pool_ops)

        data.update(one_body)

        data.update(name)

        data.update(ref_state_dict)

    print("Number of energy measurements:", ansatz.energy_calls)
    return data, nuc


class ADAPT_mixed_VQE(VQE):
    """
    Child class to define the hybrid/mixed ADAPT VQE.

    Attributes:
        ansatz (ADAPT_mixed_Ansatz): ADAPT Ansatz object.
        nucleus (Nucleus): Nucleus object.
        parameters (list): List of parameters.
        max_layers (int): Maximum number of layers.

    Methods:
        run: Runs the ADAPT VQE algorithm.
        callback: Callback function to store the energy and parameters at each iteration and stop the optimization if the threshold is
    """

    def __init__(
        self,
        data: dict,
        ansatz: ADAPT_mixed_Ansatz,
        method: str = "COBYLA",
        test_threshold: float = 1e-4,
        max_layers: int = 100,
        exact: bool = True,
    ) -> None:

        self.encoding = ansatz.encoding
        super().__init__(
            test_threshold=test_threshold, method=method, encoding=self.encoding
        )
        self.ansatz = ansatz
        self.nucleus = ansatz.nucleus
        self.parameters = []

        self.ofset = 0
        if self.encoding == "HCB":
            if self.nucleus.nucleus.qubits[0] != 0:
                if self.nucleus.shell == "p":
                    self.ofset = 0
                elif self.nucleus.shell == "sd":
                    self.ofset = 0

        self.max_layers = max_layers
        self.data = data
        self.exact = exact
        self.operators_used = data["used_operators"]

    def _op_label_args(self, op) -> tuple:
        """Returns the operator's identifying indices, as positional print args."""
        if self.encoding == "fermionic":
            return (op.ijkl,)
        return (op.A + self.ofset, op.B + self.ofset)

    def _op_label_str(self, op) -> str:
        """Returns the operator's identifying indices, formatted for an f-string (no separator, matching original HCB format)."""
        if self.encoding == "fermionic":
            return f"{op.ijkl}"
        return f"{op.A+self.ofset}{op.B+self.ofset}"

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

        print(
            " --------------------------------------------------------------------------"
        )
        print("                            ADAPT for ", self.nucleus.name)
        print(
            " --------------------------------------------------------------------------\n"
        )

        E0 = self.ansatz.energy(self.parameters)
        self.energy.append(E0)
        self.rel_error.append(
            abs((E0 - self.ansatz.nucleus.eig_val[0]) / self.ansatz.nucleus.eig_val[0])
        )

        print("Initial Energy: ", E0)
        next_operator = self.ansatz.choose_operator()

        energy_layers = [E0]
        rel_error_layers = [self.rel_error[-1]]

        max_ex = False

        while (
            self.ansatz.capas < len(self.operators_used)
            and self.ansatz.capas < self.max_layers
        ):

            self.ansatz.capas += 1

            self.ansatz.added_operators.append(next_operator)

            self.parameters.append(0.0)

            try:

                result = minimize(
                    self.ansatz.energy,
                    self.parameters,
                    method=self.method,
                    callback=self.callback,
                    options=self.options,
                )

                self.parameters = list(result.x)

                if len(self.parameters) < len(self.data["parameters"]):
                    next_operator = self.ansatz.choose_operator()
                    energy_layers.append(self.energy[-1])
                    rel_error_layers.append(self.rel_error[-1])

            except OptimizationConvergedException:
                pass
            except RuntimeError:
                print("Maximum number of executions reached")
                max_ex = True
            except Exception as e:
                print(e)
                print("FALLO: ", self.parameters)

            rel_error = abs(
                (self.energy[-1] - self.ansatz.nucleus.eig_val[0])
                / self.ansatz.nucleus.eig_val[0]
            )

            if rel_error < self.test_threshold or max_ex == True:
                self.success = True

                self.ansatz.minimum = True
                break
            print(f"\n------------ LAYER {len(energy_layers)-1} ------------")
            print("Operator:", *self._op_label_args(self.ansatz.added_operators[-1]))
            print("Energy: ", energy_layers[-1])
            print("Rel. Error: ", rel_error_layers[-1])
            print("Theta:", self.parameters[-1])

        energy_layers.append(self.energy[-1])
        rel_error_layers.append(self.rel_error[-1])
        print(f"\n------------ LAYER {len(energy_layers)-1} ------------")
        print("Operator:", *self._op_label_args(self.ansatz.added_operators[-1]))
        print("Energy: ", energy_layers[-1])
        print("Rel. Error: ", rel_error_layers[-1])
        print(
            "New operator: ",
            *self._op_label_args(self.ansatz.added_operators[-1]),
            "    Theta:",
            self.parameters[-1],
        )

        print("\nOperators used for each layer:")

        for i in range(len(self.parameters)):
            print(
                f"Layer {i+1}: Operator {self._op_label_str(self.ansatz.added_operators[i])}, Theta = {self.parameters[i]}"
            )

        print(
            f"\n Final energy result: {energy_layers[-1]}\t",
            f"Final relative error is {self.rel_error[-1]}",
        )

        if self.encoding == "fermionic":
            used_operators = [op.ijkl for op in self.ansatz.added_operators]
        else:
            used_operators = [
                [op.A + self.ofset, op.B + self.ofset]
                for op in self.ansatz.added_operators
            ]

        data = {
            "parameters": self.parameters,
            "used_operators": used_operators,
            "operator_pool": self.ansatz.operator_pool,
            "Energy": energy_layers[-1],
        }

        return data

    def callback(self, params: list) -> None:
        """
        Callback function to store the energy and parameters at each iteration and stop the optimization if the threshold is reached.
        """

        E = self.ansatz.last_energy

        self.energy.append(E)
        self.rel_error.append(
            abs((E - self.ansatz.nucleus.eig_val[0]) / self.ansatz.nucleus.eig_val[0])
        )
        self.parameters = params

        if self.rel_error[-1] <= self.test_threshold:
            self.success = True
            self.ansatz.minimum = True
            raise OptimizationConvergedException


def ADAPT_mixed_minimization(
    data: dict,
    nucleus: Nucleus,
    ref_state: int = 0,
    threshold: float = 1e-2,
    max_layers: int = 20,
    exact: bool = True,
    nshots: int = 1000,
    max_executions: int = 100,
    encoding: str = "fermionic",
    statevector_caching: bool = True,
):

    if exact:
        opt = "L-BFGS-B"
    else:
        opt = "COBYLA"

    ansatz = ADAPT_mixed_Ansatz(
        data=data,
        nucleus=nucleus,
        ref_state=ref_state,
        exact=exact,
        nshots=nshots,
        max_executions=max_executions,
        encoding=encoding,
        statevector_caching=statevector_caching,
    )

    vqe = ADAPT_mixed_VQE(
        data=data,
        ansatz=ansatz,
        method=opt,
        test_threshold=threshold,
        max_layers=max_layers,
        exact=exact,
    )

    if encoding == "fermionic":
        ham = nucleus.Ham_2_body_contributions()

        two_index = []
        ham_pool = []

        for op in ham:
            if len(set(op.ijkl)) == 2:
                two_index.append(op)
            else:
                ham_pool.append(op)

    data = vqe.run()

    if encoding == "fermionic":
        one_body, monoparticular = nucleus.Ham_1_body_contributions()
        diag_two_body_ops = {"two_index": two_index}
        ham_pool_ops = {"ham_pool": ham_pool}
        one_body = {"one_body": one_body}
        monoparticular = {"monoparticular": monoparticular}
        name = {"name": nucleus}
        ref_state_dict = {"ref_state": ref_state}

        data.update(monoparticular)

        data.update(diag_two_body_ops)

        data.update(ham_pool_ops)

        data.update(one_body)

        data.update(name)

        data.update(ref_state_dict)

    print("Number of energy measurements:", ansatz.energy_calls)
    return data, nucleus


class GGF_ADAPTVQE(VQE):
    """
    Child class to define the GGF-ADAPT VQE, for either the fermionic or the HCB
    (hard-core boson) encoding. Analytically determines the optimal parameter for
    each operator, significantly reducing the number of circuit executions.

    Attributes:
        ansatz (GGF_ADAPT_Ansatz): ADAPT Ansatz object.
        nucleus (Nucleus): Nucleus object.
        parameters (list): List of parameters.
        max_layers (int): Maximum number of layers.

    Methods:
        run: Runs the ADAPT VQE algorithm.
        callback: Callback function to store the energy and parameters at each iteration and stop the optimization if the threshold is
    """

    def __init__(
        self,
        ansatz: GGF_ADAPT_Ansatz,
        method: str = "L-BFGS-B",
        conv_criterion: str = "Repeated op",
        test_threshold: float = 1e-2,
        max_layers: int = 100,
    ) -> None:

        self.encoding = ansatz.encoding
        super().__init__(
            test_threshold=test_threshold, method=method, encoding=self.encoding
        )
        self.ansatz = ansatz
        self.nucleus = ansatz.nucleus

        self.ofset = 0
        if self.encoding == "HCB":
            if self.nucleus.nucleus.qubits[0] != 0:
                if self.nucleus.shell == "p":
                    self.ofset = 0
                elif self.nucleus.shell == "sd":
                    self.ofset = 0

        self.parameters = []
        self.max_layers = max_layers

        self.conv_criterion = conv_criterion

    def _op_label_args(self, op) -> tuple:
        """Returns the operator's identifying indices, as positional print args."""
        if self.encoding == "fermionic":
            return (op.ijkl,)
        return (op.A + self.ofset, op.B + self.ofset)

    def _op_label_str(self, op) -> str:
        """Returns the operator's identifying indices, formatted for an f-string."""
        if self.encoding == "fermionic":
            return f"{op.ijkl}"
        return f"{op.A+self.ofset},{op.B+self.ofset}"

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

        print(
            " --------------------------------------------------------------------------"
        )
        print("                            ADAPT for ", self.nucleus.name)
        print(
            " --------------------------------------------------------------------------\n"
        )

        E0 = self.ansatz.energy(self.parameters, save_ansatz=True)

        rel_error = abs(
            (E0 - self.ansatz.nucleus.eig_val[0]) / self.ansatz.nucleus.eig_val[0]
        )
        self.ansatz.set_optimized_state([], E0)
        print("Initial Energy: ", E0)

        next_operator, min_energy, best_parameter = self.ansatz.choose_operator()

        energy_layers = [E0]
        rel_error_layers = [rel_error]

        max_ex = False

        while (
            self.ansatz.minimum == False
            and len(self.ansatz.added_operators) < self.max_layers
        ):
            self.ansatz.added_operators.append(next_operator)
            self.parameters.append(best_parameter)
            current_layer_params = self.parameters
            min_energy = self.ansatz.energy(current_layer_params, save_ansatz=True)

            current_layer_energy = min_energy
            rel_error = abs(
                (current_layer_energy - self.ansatz.nucleus.eig_val[0])
                / self.ansatz.nucleus.eig_val[0]
            )
            self.ansatz.set_optimized_state(current_layer_params, current_layer_energy)

            try:
                if len(self.parameters) > 1:

                    result = minimize(
                        self.ansatz.energy,
                        self.parameters,
                        method=self.method,
                        callback=self.callback,
                        options=self.options,
                    )
                    self.parameters = list(result.x)

                    current_layer_energy = self.ansatz.global_best_energy
                    current_layer_params = self.ansatz.global_best_params

                    energy_layers.append(current_layer_energy)
                    rel_error_layers.append(rel_error)

                if len(self.ansatz.added_operators) < self.max_layers:

                    next_operator, min_energy, best_parameter = (
                        self.ansatz.choose_operator()
                    )

                    if next_operator == self.ansatz.added_operators[-1]:
                        self.ansatz.minimum = True
                    elif abs(energy_layers[-1] - min_energy) < self.test_threshold:
                        self.ansatz.minimum = True
                    else:
                        energy_layers.append(current_layer_energy)
                        rel_error_layers.append(rel_error)

            except OptimizationConvergedException:
                current_layer_energy = self.ansatz.global_best_energy
                current_layer_params = self.ansatz.global_best_params
                energy_layers.append(current_layer_energy)
                self.ansatz.set_optimized_state(
                    current_layer_params, current_layer_energy
                )
                rel_error_layers.append(rel_error)

                pass
            except RuntimeError:
                print("Maximum number of executions reached")
                max_ex = True
            except Exception as e:
                print(e)
                print("FALLO: ", self.parameters)

            self.ansatz.set_optimized_state(current_layer_params, current_layer_energy)
            rel_error = abs(
                (current_layer_energy - self.ansatz.nucleus.eig_val[0])
                / self.ansatz.nucleus.eig_val[0]
            )

            if (
                rel_error < self.test_threshold
                or len(self.parameters) == self.max_layers
                or max_ex == True
            ):
                self.success = True
                self.ansatz.minimum = True
                break

            print(f"\n------------ LAYER {len(self.parameters)} ------------")
            print("Operator:", *self._op_label_args(self.ansatz.added_operators[-1]))
            print("Energy: ", energy_layers[-1])
            print("Rel. Error: ", rel_error)
            print("Theta:", self.parameters)

        print(f"\n------------ LAYER {len(self.parameters)} ------------")
        print("Operator:", *self._op_label_args(self.ansatz.added_operators[-1]))
        print("Energy: ", current_layer_energy)
        print("Rel. Error: ", rel_error_layers[-1])
        print("Theta:", self.parameters)

        print("\nOperators used for each layer:")
        for i, op in enumerate(self.ansatz.added_operators):
            print(
                f"Layer {i}: Operator {self._op_label_str(op)}, Theta = {self.parameters[i]}"
            )

        print(
            f"\n Final energy result: {energy_layers[-1]}\t",
            f"Final relative error is {rel_error}",
        )

        print(energy_layers)

        if self.conv_criterion == "None" and self.ansatz.minimum == False:
            self.ansatz.minimum = True

        data = {
            "parameters": self.parameters,
            "used_operators": [op for op in self.ansatz.added_operators],
            "operator_pool": self.ansatz.operator_pool,
            "Energy": energy_layers,
            "n_measurements": self.ansatz.energy_calls,
        }

        return data

    def callback(self, params: list) -> None:
        """
        Callback function to store the energy and parameters at each iteration and stop the optimization if the threshold is reached.
        """

        E = self.ansatz.global_best_energy

        rel_error = abs(
            (E - self.ansatz.nucleus.eig_val[0]) / self.ansatz.nucleus.eig_val[0]
        )

        if rel_error < self.test_threshold:
            self.success = True
            self.ansatz.minimum = True
            raise OptimizationConvergedException


def GGF_ADAPT_minimization(
    nucleus: str,
    ref_state: int = 0,
    opt_method: str = "L-BFGS-B",
    threshold: float = 1e-2,
    max_layers: int = 20,
    shell: str = "p",
    encoding: str = "fermionic",
):

    nuc = Nucleus(nucleus, shell=shell, encoding=encoding)
    ref_state = np.eye(nuc.d_H)[ref_state]

    ansatz = GGF_ADAPT_Ansatz(nucleus=nuc, ref_state=ref_state, encoding=encoding)

    vqe = GGF_ADAPTVQE(
        ansatz=ansatz,
        method=opt_method,
        test_threshold=threshold,
        max_layers=max_layers,
    )

    data = vqe.run()

    print("Number of energy measurements:", ansatz.energy_calls)
    return data, nuc


def Quantum_GGF_ADAPT_minimization(
    nucleus: str,
    ref_state: int = 0,
    threshold: float = 1e-2,
    max_layers: int = 20,
    shell: str = "p",
    exact: bool = True,
    nshots: int = 1000,
    max_executions: int = 1000,
    encoding: str = "fermionic",
    statevector_caching: bool = True,
):

    nuc = Nucleus(nucleus, shell=shell, encoding=encoding)

    ansatz = Quantum_GGF_ADAPTAnsatz(
        nucleus=nuc,
        ref_state=ref_state,
        exact=exact,
        nshots=nshots,
        max_executions=max_executions,
        encoding=encoding,
        statevector_caching=statevector_caching,
    )

    if exact:
        opt = "L-BFGS-B"
    else:
        opt = "COBYLA"

    vqe = GGF_ADAPTVQE(
        ansatz=ansatz, method=opt, test_threshold=threshold, max_layers=max_layers
    )

    data = vqe.run()

    print("Number of energy measurements:", ansatz.energy_calls)

    return data, nuc


class GGF_VQE(VQE):
    """
    Child class to define the GGF VQE with a fixed operator pool, for either the
    fermionic or the HCB (hard-core boson) encoding.

    Attributes:
        ansatz (GGF_Ansatz): ADAPT Ansatz object.
        nucleus (Nucleus): Nucleus object.
        parameters (list): List of parameters.
        max_layers (int): Maximum number of layers.

    Methods:
        run: Runs the ADAPT VQE algorithm.
    """

    def __init__(
        self,
        ansatz: GGF_Ansatz,
        method: str = "L-BFGS-B",
        conv_criterion: str = "Repeated op",
        test_threshold: float = 1e-2,
        max_layers: int = 100,
    ) -> None:

        self.encoding = ansatz.encoding
        super().__init__(
            test_threshold=test_threshold, method=method, encoding=self.encoding
        )
        self.ansatz = ansatz
        self.nucleus = ansatz.nucleus

        self.ofset = 0
        if self.encoding == "HCB":
            if self.nucleus.nucleus.qubits[0] != 0:
                if self.nucleus.shell == "p":
                    self.ofset = 0
                elif self.nucleus.shell == "sd":
                    self.ofset = 0

        self.parameters = []
        self.max_layers = max_layers

        try:
            self.conv_criterion = conv_criterion
        except conv_criterion not in ["Repeated op", "Gradient", "None"]:
            print(
                'Invalid minimum criterion. Choose between "Repeated op", "Gradient" and "None"'
            )
            exit()

    def _op_label_args(self, op) -> tuple:
        """Returns the operator's identifying indices, as positional print args."""
        if self.encoding == "fermionic":
            return (op.ijkl,)
        return (op.A + self.ofset, op.B + self.ofset)

    def _op_label_str(self, op) -> str:
        """Returns the operator's identifying indices, formatted for an f-string."""
        if self.encoding == "fermionic":
            return f"{op.ijkl}"
        return f"{op.A+self.ofset},{op.B+self.ofset}"

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

        print(
            " --------------------------------------------------------------------------"
        )
        print("                            ADAPT for ", self.nucleus.name)
        print(
            " --------------------------------------------------------------------------\n"
        )

        E0 = self.ansatz.energy(self.parameters, save_ansatz=True)
        self.energy.append(E0)
        rel_error = abs(
            (E0 - self.ansatz.nucleus.eig_val[0]) / self.ansatz.nucleus.eig_val[0]
        )

        self.ansatz.set_optimized_state([], E0)
        print("Initial Energy: ", E0)
        next_operator, min_energy, best_parameter = self.ansatz.choose_operator()

        energy_layers = [E0]
        rel_error_layers = [rel_error]

        max_ex = False

        while (
            self.ansatz.minimum == False
            and len(self.ansatz.added_operators) < self.max_layers
        ):

            self.ansatz.added_operators.append(next_operator)
            self.parameters.append(best_parameter)

            current_layer_params = self.parameters

            min_energy = self.ansatz.energy(current_layer_params, save_ansatz=True)

            current_layer_energy = min_energy
            rel_error = abs(
                (current_layer_energy - self.ansatz.nucleus.eig_val[0])
                / self.ansatz.nucleus.eig_val[0]
            )

            self.ansatz.set_optimized_state(current_layer_params, current_layer_energy)
            print(current_layer_energy, current_layer_params)

            if len(self.ansatz.added_operators) < self.max_layers:

                next_operator, min_energy, best_parameter = (
                    self.ansatz.choose_operator()
                )

                if abs(energy_layers[-1] - min_energy) < self.test_threshold:
                    self.ansatz.minimum = True
                    energy_layers.append(current_layer_energy)
                    rel_error_layers.append(rel_error)

                else:
                    energy_layers.append(current_layer_energy)
                    rel_error_layers.append(rel_error)

            self.ansatz.set_optimized_state(current_layer_params, current_layer_energy)
            rel_error = abs(
                (current_layer_energy - self.ansatz.nucleus.eig_val[0])
                / self.ansatz.nucleus.eig_val[0]
            )

            if (
                rel_error < self.test_threshold
                or len(self.parameters) == self.max_layers
                or max_ex == True
            ):

                energy_layers.append(current_layer_energy)
                rel_error_layers.append(rel_error)
                self.success = True
                self.ansatz.minimum = True
                break

            print(f"\n------------ LAYER {len(self.parameters)} ------------")
            print("Operator:", *self._op_label_args(self.ansatz.added_operators[-1]))
            print("Energy: ", energy_layers[-1])
            print("Rel. Error: ", rel_error)
            print("Theta:", self.parameters[-1])

        print(f"\n------------ LAYER {len(self.parameters)} ------------")
        print("Operator:", *self._op_label_args(self.ansatz.added_operators[-1]))
        print("Energy: ", current_layer_energy)
        print("Rel. Error: ", rel_error_layers[-1])
        print("Theta:", current_layer_params[-1])

        print("\nOperators used for each layer:")
        for i, op in enumerate(self.ansatz.added_operators):
            print(
                f"Layer {i}: Operator {self._op_label_str(op)}, Theta = {self.parameters[i]}"
            )

        print(
            f"\n Final energy result: {energy_layers[-1]}\t",
            f"Final relative error is {rel_error}",
        )

        print(energy_layers)

        if self.conv_criterion == "None" and self.ansatz.minimum == False:
            self.ansatz.minimum = True

        data = {
            "parameters": self.parameters,
            "used_operators": [op for op in self.ansatz.added_operators],
            "operator_pool": self.ansatz.operator_pool,
            "Energy": energy_layers[-1],
            "n_measurements": self.ansatz.energy_calls,
        }

        return data


def GGF_minimization(
    nucleus: str,
    ref_state: int = 0,
    opt_method: str = "L-BFGS-B",
    threshold: float = 1e-2,
    max_layers: int = 20,
    shell: str = "p",
    encoding: str = "fermionic",
):

    nuc = Nucleus(nucleus, shell=shell, encoding=encoding)
    ref_state = np.eye(nuc.d_H)[ref_state]

    ansatz = GGF_Ansatz(nucleus=nuc, ref_state=ref_state, encoding=encoding)

    vqe = GGF_VQE(
        ansatz=ansatz,
        method=opt_method,
        test_threshold=threshold,
        max_layers=max_layers,
    )

    data = vqe.run()

    print("Number of energy measurements:", ansatz.energy_calls)
    return data, nuc


def Quantum_GGF_minimization(
    nucleus: str,
    ref_state: int = 0,
    threshold: float = 1e-2,
    max_layers: int = 20,
    shell: str = "p",
    exact: bool = True,
    nshots: int = 1000,
    max_executions: int = 1000,
    encoding: str = "fermionic",
    statevector_caching: bool = True,
):

    nuc = Nucleus(nucleus, shell=shell, encoding=encoding)

    ansatz = Quantum_GGF_Ansatz(
        nucleus=nuc,
        ref_state=ref_state,
        exact=exact,
        nshots=nshots,
        max_executions=max_executions,
        encoding=encoding,
        statevector_caching=statevector_caching,
    )

    if exact:
        opt = "L-BFGS-B"
    else:
        opt = "COBYLA"

    vqe = GGF_VQE(
        ansatz=ansatz, method=opt, test_threshold=threshold, max_layers=max_layers
    )

    data = vqe.run()

    print("Number of energy measurements:", ansatz.energy_calls)

    return data, nuc

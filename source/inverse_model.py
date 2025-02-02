from types import MappingProxyType
import source.flow_network as flow_network
import source.inverseproblemmodules.adjoint_method_implementations as adj_method_parameters
import source.inverseproblemmodules.adjoint_method_solver as adj_method_solver
import source.inverseproblemmodules.alpha_restriction as alpha_mapping
import source.fileio.read_target_values as read_target_values
import source.fileio.read_parameters as read_parameters


class InverseModel(object):
    # todo docstring and explain all attributes
    def __init__(self, flownetwork: flow_network.FlowNetwork, imp_readtargetvalues: read_target_values.ReadTargetValues,
                 imp_readparameters: read_parameters.ReadParameters,
                 imp_adjointmethodparameters: adj_method_parameters.AdjointMethodImplementations,
                 imp_adjointmethodsolver: adj_method_solver.AdjointMethodSolver,
                 imp_alphamapping: alpha_mapping.AlphaRestriction, PARAMETERS: MappingProxyType):
        # "Reference" to flow network
        self._flow_network = flownetwork

        # "Reference" to parameter dict
        self._PARAMETERS = PARAMETERS

        # Target values
        self.edge_constraint_eid = None
        self.edge_constraint_type = None # 1: Flow rate, 2: Velocity, ...
        self.edge_constraint_value = None
        self.edge_constraint_range_pm = None
        self.edge_constraint_sigma = None
        self.nr_of_edge_constraints = None

        # Parameter space
        # Edge parameters
        self.edge_param_eid = None
        self.parameter_pm_range = None
        self.nr_of_edge_parameters = None

        # Vertex parameters
        self.vertex_param_vid = None
        # self.vertex_param_pm_range = None
        self.nr_of_vertex_parameters = None

        # Total parameters
        self.nr_of_parameters = None

        # Parameter edge and vertex attributes
        self.alpha = None
        self.alpha_prime = None
        self.alpha_pm_range = None

        self.transmiss_baselinevalue = None
        self.diameter_baselinevalue = None

        self.boundary_pressure_baselinevalue = None

        self.mu_rel_tilde = None
        self.transmiss_tilde = None

        # Inverse model parameters
        self.gamma = None
        self.phi = None

        # Inverse model cost terms
        self.f_h = None  # Cost of hard constraint

        # Adjoint method vectors and matrices
        self.d_f_d_alpha = None  # Vector
        self.d_f_d_pressure = None  # Vector
        self.d_g_d_alpha = None  # coo_matrix

        # Gradient
        self.gradient_alpha = None
        self.gradient_alpha_prime = None

        # "References" to implementations
        self._imp_adjointmethodparameters = imp_adjointmethodparameters
        self._imp_readtargetvalues = imp_readtargetvalues
        self._imp_readparameters = imp_readparameters
        self._imp_adjointmethodsolver = imp_adjointmethodsolver
        self._imp_alphamapping = imp_alphamapping

    def initialise_inverse_model(self):
        """
        Method to initialise the inverse model based on target values and defined parameters.
        """
        self._imp_readtargetvalues.read(self, self._flow_network)
        self._imp_readparameters.read(self, self._flow_network)
        self._imp_adjointmethodparameters.initialise_parameters(self, self._flow_network)
        self.gamma = self._PARAMETERS["gamma"]
        self.phi = self._PARAMETERS["phi"]

    def update_state(self):
        """
        Method to update the parameter vector alpha with constant step-width gradient descent.
        """
        # Update all partial derivatives needed to solve the adjoint method
        self._imp_adjointmethodparameters.update_partial_derivatives(self, self._flow_network)
        # Update gradient d f / d alpha by solving the adjoint method
        self._imp_adjointmethodsolver.update_gradient_alpha(self, self._flow_network)
        # Update gradient d f / f alpha_prime (mapping between parameter and pseudo parameter)
        self._imp_alphamapping.update_gradient_alpha_prime(self)
        # Update alpha_prime by using gradient descent with constant learning rate.
        # Todo: different algorithms, e.g. with adaptive gamma or Adams algorithm
        self.alpha_prime -= self.gamma * self.gradient_alpha_prime
        # Transform pseudo parameter alpha_prime back to alpha space
        self._imp_alphamapping.update_alpha_from_alpha_prime(self)
        # Update the system state depending on the parameter (e.g. diameter, transmissibility, boundary pressures)
        self._imp_adjointmethodparameters.update_state(self, self._flow_network)

    def update_cost(self):
        """
        Method to update the cost function value
        """
        # Update the cost function value
        self._imp_adjointmethodparameters.update_cost_hardconstraint(self, self._flow_network)
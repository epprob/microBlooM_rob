import sys
from types import MappingProxyType
from source.inverse_model import InverseModel
import source.flow_network as flow_network
import source.inverseproblemmodules.adjoint_method_implementations as adj_method_parameters
import source.inverseproblemmodules.adjoint_method_solver as adj_method_solver
import source.inverseproblemmodules.alpha_restriction as alpha_mapping
import source.fileio.read_target_values as read_target_values
import source.fileio.read_parameters as read_parameters
import source_sensitivity.adjoint_sensitivity_implementations as  adjoint_sensitivity_implementations
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import norm


class InverseModelSensitivity(InverseModel):

    def __init__(self, flownetwork: flow_network.FlowNetwork, imp_readtargetvalues: read_target_values.ReadTargetValues,
                 imp_readparameters: read_parameters.ReadParameters,
                 imp_adjointmethodparameters: adj_method_parameters.AdjointMethodImplementations,
                 imp_adjointmethodsolver: adj_method_solver.AdjointMethodSolver,
                 imp_alphamapping: alpha_mapping.AlphaRestriction, PARAMETERS: MappingProxyType):
        super().__init__(flownetwork, imp_readtargetvalues, imp_readparameters, imp_adjointmethodparameters,
                         imp_adjointmethodsolver, imp_alphamapping, PARAMETERS)

        self.param_sensitivity = None
        self.sensitivity_matrix = None
        self.d_flowrates_d_pressure = None
        self.d_flowrates_d_alpha = None

    def initialise_inverse_model(self):
        self._imp_readparameters.read(self, self._flow_network)

    def update_state(self):
        pass

    def update_cost(self):
        pass

    def update_sensitivity(self):

        adjoint_sensitivity_implementations.update_d_flowrateall_d_alpha(self, self._flow_network)
        adjoint_sensitivity_implementations.update_d_flowrateall_d_pressure(self, self._flow_network)
        self._imp_adjointmethodparameters._update_d_g_d_alpha(self, self._flow_network)

        lambda_matrix = spsolve(csc_matrix(self._flow_network.system_matrix.transpose()),
                                -csc_matrix(self.d_flowrates_d_pressure.transpose()))

        self.sensitivity_matrix = lambda_matrix.transpose().dot(csc_matrix(self.d_g_d_alpha)) + self.d_flowrates_d_alpha

        self.param_sensitivity = norm(self.sensitivity_matrix, axis=1)

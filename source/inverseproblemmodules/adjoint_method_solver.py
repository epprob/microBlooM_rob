from abc import ABC, abstractmethod
from types import MappingProxyType
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


class AdjointMethodSolver(ABC):
    """
    Abstract base class for updating the gradient in alpha space
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of PressureFlowSolver.
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS

    @abstractmethod
    def _get_lambda_vector(self, inversemodel, flownetwork):
        """
        Abstract method to solve the adjoint equation.
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        :returns: Lambda vector
        :rtype: 1d numpy array
        """

    def update_gradient_alpha(self, inversemodel, flownetwork):
        # Compute the vector lambda by solving the adjoint equation
        lambda_vector = self._get_lambda_vector(inversemodel, flownetwork)
        # Compute the gradient (1d array).
        # Note: this is identical to lambda^T*dg/dalpha + df/dalpha, but should ensure that calculation remains sparse.
        gradient = inversemodel.d_g_d_alpha.transpose().dot(lambda_vector)+inversemodel.d_f_d_alpha
        # Update the gradient with respect to alpha.
        inversemodel.gradient_alpha = gradient

class AdjointMethodSolverSparseDirect(AdjointMethodSolver):
    """
    Class for updating the gradient in alpha space with a sparse direct solver.
    """

    def _get_lambda_vector(self, inversemodel, flownetwork):
        """
        Solve the adjoint equation with a sparse direct solver.
        :param inversemodel: inverse model object
        :type inversemodel: source.inverse_model.InverseModel
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        :returns: Lambda vector
        :rtype: 1d numpy array
        """
        lambda_vector = spsolve(csc_matrix(flownetwork.system_matrix.transpose()), -inversemodel.d_f_d_pressure)
        return lambda_vector

# Todo: Implement other solvers (cg, amg, ...), do when necessary
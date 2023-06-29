from scipy.sparse import coo_matrix
import numpy as np


def update_d_flowrateall_d_alpha(inversemodel, flownetwork):
    # Basically the zero matrix
    inversemodel.d_flowrates_d_alpha = coo_matrix((flownetwork.nr_of_es, inversemodel.nr_of_parameters),
                                                  dtype=float)


def update_d_flowrateall_d_pressure(inversemodel, flownetwork):
    # Partial derivative of flow rate with respect to all pressures
    edge_list = flownetwork.edge_list

    transmissibilities = flownetwork.transmiss

    edge_indices = np.arange(flownetwork.nr_of_es)

    row = np.append(edge_indices, edge_indices)
    col = np.append(edge_list[:, 0], edge_list[:, 1])
    data = np.append(transmissibilities, -transmissibilities)  # for flow rate sensitivity

    # for velocity based sensitivity (hack
    # diameters = flownetwork.diameter
    # data = np.append(transmissibilities * 4. / (np.square(diameters) * np.pi),
    #                  -transmissibilities * 4. / (np.square(diameters) * np.pi))

    inversemodel.d_flowrates_d_pressure = coo_matrix((data, (row, col)),
                                                     shape=(flownetwork.nr_of_es, flownetwork.nr_of_vs))
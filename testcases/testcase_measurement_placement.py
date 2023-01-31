"""
A python script to estimate edge parameters such as diameters and transmissibilities of microvascular networks based
on given flow rates and velocities in selected edges. Capabilities:
1. Import a network from file or generate a hexagonal network
2. Compute the edge transmissibilities with taking the impact of RBCs into account (Fahraeus, Fahraeus-Linquist effects)
3. Solve for flow rates, pressures and RBC velocities
4. Update the boundary pressures with a gradient descent algorithm minimising a given cost function.
5. Restriction of parameters to desired ranges (target value +/- tolerance).
6. Individual selection of parameter boundary vertices and target edges.
7. Target flow rates and velocities can be specified and combined into a single cost function.
8. Tuning of absolute boundary pressures.
9. Optimisation of pressures for a fixed number of iteration steps.
10. Save the results in a file.
"""

from source.flow_network import FlowNetwork
from source_sensitivity.inverse_model_sensitivity import InverseModelSensitivity
from types import MappingProxyType
import source.setup.setup as setup
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


# MappingProxyType is basically a const dict
PARAMETERS = MappingProxyType(
    {
        # Setup parameters for blood flow model
        "read_network_option": 1,  # 1: generate hexagonal graph
                                   # 2: import graph from csv files
                                   # 3: todo import graph from igraph files
                                   # 4: todo import graph from edge_data and vertex_data pickle files
        "write_network_option": 1,  # 1: do not write anything
                                    # 2: igraph format
                                    # 3-...: todo other file formats. also handle overwrite data, etc
        "tube_haematocrit_option": 1,  # 1: No RBCs (ht=0)
                                       # 2: Constant haematocrit
                                       # 3: todo: RBC tracking
                                       # 4-xxx: todo: steady state RBC laws
        "rbc_impact_option": 1,  # 1: hd = ht (makes only sense if tube_haematocrit_option:1 or ht=0)
                                 # 2: Laws by Pries, Neuhaus, Gaehtgens (1992)
                                 # 3: todo Other laws. in vivo?
        "solver_option": 1,  # 1: Direct solver
                             # 2: todo: other solvers (CG, AMG, ...)
        # Blood properties
        "ht_constant": 0.0,
        "mu_plasma": 0.0012,
        # Hexagonal network properties
        "nr_of_hexagon_x": 5,
        "nr_of_hexagon_y": 5,
        "hexa_edge_length": 62.e-6,
        "hexa_diameter": 4.e-6,
        "hexa_boundary_vertices": [0, 65, 21, 48, 41, 31, 56],
        "hexa_boundary_values": [2, 1, 3, 1, 1, 5, 2],
        "hexa_boundary_types": [1, 1, 1, 1, 1, 1, 1],  # 1: pressure, 2: flow rate
        # Import network from csv options
        "csv_path_vertex_data": "data/network/b6_B_pre_061/node_data.csv",
        "csv_path_edge_data": "data/network/b6_B_pre_061/edge_data.csv",
        "csv_path_boundary_data": "data/network/b6_B_pre_061/boundary_node_data.csv",
        "csv_diameter": "D", "csv_length": "L",
        "csv_edgelist_v1": "n1", "csv_edgelist_v2": "n2",
        "csv_coord_x": "x", "csv_coord_y": "y", "csv_coord_z": "z",
        "csv_boundary_vs": "nodeId", "csv_boundary_type": "boundaryType", "csv_boundary_value": "boundaryValue",
        # Write options
        "write_override_initial_graph": False,
        "write_path_igraph": "data/network/b6_B_pre_061_simulated.pkl", # only required for "write_network_option" 2
        ##########################
        # Inverse problem options
        # Define parameter space
        "parameter_space": 11,  # 1: Relative diameter to baseline (alpha = d/d_base)
                                # 2: Relative transmissibility to baseline (alpha = T/T_base)
                                # 11: Pressure boundary condition values (alpha = p_0)
        "parameter_restriction": 1,  # 1: No restriction of parameter values (alpha_prime = alpha)
                                     # 2: Restriction of parameter by a +/- tolerance to baseline
        "inverse_model_solver": 1,  # Direct solver
                                    # 2: todo: other solvers (CG, AMG, ...)
        # Filepath to prescribe target values / constraints on edges
        "csv_path_edge_target_data": "data/inverse_model/edge_target_BC_tuning.csv",
        # Filepath to define the edge parameter space (only for tuning of diameters and transmissibilities)
        "csv_path_edge_parameterspace": "not needed",
        # Filepath to define the vertex parameter space (only for tuning of boundary conditions)
        "csv_path_vertex_parameterspace": "data/inverse_model/vertex_parameters.csv",
        # Gradient descent options:
        "gamma": .1,
        "phi": .5,  # for parameter_restriction 2
        "max_nr_of_iterations": 50  # Maximum of iterations
    }
)

setup_simulation = setup.SetupSimulation()

# Initialise objects related to simulate blood flow without RBC tracking.
imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_velocity, imp_buildsystem, \
    imp_solver = setup_simulation.setup_bloodflow_model(PARAMETERS)

# Initialise objects related to the inverse model.
imp_readtargetvalues, imp_readparameters, imp_adjoint_parameter, imp_adjoint_solver, \
    imp_alpha_mapping = setup_simulation.setup_inverse_model(PARAMETERS)

# Initialise objects related to the inverse model.

# Initialise flownetwork and inverse model objects
flow_network_ground_truth = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                                        imp_solver, imp_velocity, PARAMETERS)
inverse_model_sensitivity = InverseModelSensitivity(flow_network_ground_truth, imp_readtargetvalues, imp_readparameters, imp_adjoint_parameter,
                                                    imp_adjoint_solver, imp_alpha_mapping, PARAMETERS)

print("Read network: ...")
flow_network_ground_truth.read_network()
print("Read network: DONE")

print("Update transmissibility: ...")
flow_network_ground_truth.update_transmissibility()
print("Update transmissibility: DONE")

flow_network_ground_truth.transmiss = flow_network_ground_truth.transmiss * 0. + 1.
flow_network_ground_truth.diameter = np.power(128. / np.pi * flow_network_ground_truth.transmiss * flow_network_ground_truth.length * flow_network_ground_truth.mu_rel * PARAMETERS["mu_plasma"], .25)

flow_network_ground_truth.update_transmissibility()

print("Update flow, pressure and velocity: ...")
flow_network_ground_truth.update_blood_flow()
print("Update flow, pressure and velocity: DONE")

inverse_model_sensitivity.initialise_inverse_model()

inverse_model_sensitivity.update_sensitivity()

q_sensitivities = inverse_model_sensitivity.param_sensitivity

# Plot network
xy = flow_network_ground_truth.xyz[:, :2]
edgelist = flow_network_ground_truth.edge_list

segments = xy[edgelist]

fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(xy[:,0].min()-.5*PARAMETERS['hexa_edge_length'], xy[:,0].max()+PARAMETERS['hexa_edge_length'])
ax.set_ylim(xy[:,1].min()-.5*PARAMETERS['hexa_edge_length'], xy[:,1].max()+PARAMETERS['hexa_edge_length'])
line_segments = LineCollection(segments, cmap=plt.get_cmap("viridis"), norm=plt.Normalize(vmin=0, vmax=np.max(q_sensitivities)))
line_segments.set_array(q_sensitivities)
line_segments.set_linewidth(5)
ax.add_collection(line_segments)

ax.set_title('Sensitivity wrt boundary pressures')

ax.plot(xy[flow_network_ground_truth.boundary_vs,0], xy[flow_network_ground_truth.boundary_vs,1], 'o', color='r', markersize=10)

cbar1 = plt.colorbar(line_segments)

fig.savefig("output/sensitivity.png", dpi=200)

print(np.max(q_sensitivities))





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
from source.inverse_model import InverseModel
from source_sensitivity.inverse_model_sensitivity import InverseModelSensitivity
from types import MappingProxyType
import source.setup.setup as setup
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import copy


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
        "gamma": .05,
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

# Ground truth and sensititvity
flow_network = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                           imp_solver, imp_velocity, PARAMETERS)

flow_network_sensitivity = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                           imp_solver, imp_velocity, PARAMETERS)

print("Read network: ...")
flow_network.read_network()
flow_network_sensitivity.read_network()
print("Read network: DONE")

print("Update transmissibility: ...")
flow_network.update_transmissibility()
flow_network_sensitivity.update_transmissibility()
print("Update transmissibility: DONE")

flow_network.transmiss = flow_network.transmiss * 0. + 1.
flow_network.diameter = np.power(128. / np.pi * flow_network.transmiss * flow_network.length * flow_network.mu_rel * PARAMETERS["mu_plasma"], .25)
flow_network_sensitivity.transmiss = flow_network.transmiss * 0. + 1.
flow_network_sensitivity.diameter = np.power(128. / np.pi * flow_network.transmiss * flow_network.length * flow_network.mu_rel * PARAMETERS["mu_plasma"], .25)

flow_network.update_transmissibility()
flow_network_sensitivity.update_transmissibility()

print("Update flow, pressure and velocity: ...")
flow_network.update_blood_flow()
flow_network_sensitivity.update_blood_flow()
print("Update flow, pressure and velocity: DONE")

global_eids = np.arange(flow_network.nr_of_es)
sensitivity_order = np.array([], dtype=int)

# Start with sensitivity calculations

inverse_model_sensitivity = InverseModelSensitivity(flow_network_sensitivity, imp_readtargetvalues, imp_readparameters, imp_adjoint_parameter,
                                                    imp_adjoint_solver, imp_alpha_mapping, PARAMETERS)

inverse_model_sensitivity.initialise_inverse_model()
inverse_model_sensitivity.update_sensitivity()

inverse_model_sensitivity_base = InverseModelSensitivity(flow_network_sensitivity, imp_readtargetvalues, imp_readparameters, imp_adjoint_parameter,
                                                    imp_adjoint_solver, imp_alpha_mapping, PARAMETERS)

inverse_model_sensitivity_base.initialise_inverse_model()
inverse_model_sensitivity_base.update_sensitivity()

# edge_id_remove_local = np.argmax(inverse_model_sensitivity.param_sensitivity)
# edge_id_remove_global = global_eids[edge_id_remove_local]
# sensitivity_order = np.append(sensitivity_order, edge_id_remove_global)
edge_id_remove_local = -1

for i in range(10):
    if np.size(sensitivity_order) > 0 and edge_id_remove_local >=0:

        current_remove = edge_id_remove_local

        flow_network_sensitivity.edge_list = np.delete(flow_network_sensitivity.edge_list, current_remove, axis=0)
        flow_network_sensitivity.diameter = np.delete(flow_network_sensitivity.diameter, current_remove)
        flow_network_sensitivity.length = np.delete(flow_network_sensitivity.length, current_remove)
        flow_network_sensitivity.ht = np.delete(flow_network_sensitivity.ht, current_remove)
        flow_network_sensitivity.hd = np.delete(flow_network_sensitivity.hd, current_remove)
        flow_network_sensitivity.mu_rel = np.delete(flow_network_sensitivity.mu_rel, current_remove)
        flow_network_sensitivity.transmiss = np.delete(flow_network_sensitivity.transmiss, current_remove)
        flow_network_sensitivity.flow_rate = np.delete(flow_network_sensitivity.flow_rate, current_remove)
        flow_network_sensitivity.rbc_velocity = np.delete(flow_network_sensitivity.rbc_velocity, current_remove)
        flow_network_sensitivity.nr_of_es = flow_network_sensitivity.nr_of_es - 1
        global_eids = np.delete(global_eids, current_remove)

        flow_network_sensitivity.update_linear_system()

        inverse_model_sensitivity = InverseModelSensitivity(flow_network_sensitivity, imp_readtargetvalues, imp_readparameters, imp_adjoint_parameter,
                                                            imp_adjoint_solver, imp_alpha_mapping, PARAMETERS)

        inverse_model_sensitivity.initialise_inverse_model()
        inverse_model_sensitivity.update_sensitivity()


    edge_id_remove_local = np.argmax(inverse_model_sensitivity.param_sensitivity)
    edge_id_remove_global = global_eids[edge_id_remove_local]
    sensitivity_order = np.append(sensitivity_order, edge_id_remove_global)

    # Plot network
    q_sensitivities = inverse_model_sensitivity.param_sensitivity
    xy = flow_network_sensitivity.xyz[:, :2]
    edgelist = flow_network_sensitivity.edge_list

    segments = xy[edgelist]

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(xy[:, 0].min() - .5 * PARAMETERS['hexa_edge_length'], xy[:, 0].max() + PARAMETERS['hexa_edge_length'])
    ax.set_ylim(xy[:, 1].min() - .5 * PARAMETERS['hexa_edge_length'], xy[:, 1].max() + PARAMETERS['hexa_edge_length'])
    line_segments = LineCollection(segments, cmap=plt.get_cmap("viridis"),
                                   norm=plt.Normalize(vmin=0, vmax=np.max(q_sensitivities)))
    line_segments.set_array(q_sensitivities)
    line_segments.set_linewidth(5)
    ax.add_collection(line_segments)

    arg_sensitivity_sort = np.flip(np.argsort(inverse_model_sensitivity.param_sensitivity)[-8:])

    for curr_eid in range(flow_network_sensitivity.nr_of_es):
        v_l = edgelist[curr_eid, 0]
        v_r = edgelist[curr_eid, 1]
        x = .5 * (xy[v_l, 0] + xy[v_r, 0])
        y = .5 * (xy[v_l, 1] + xy[v_r, 1])
        ax.text(x, y, curr_eid, size=8, c='aqua', ha='center', va='center')

    for curr_vid, bid in zip(flow_network_sensitivity.boundary_vs,
                             range(np.size(flow_network_sensitivity.boundary_vs))):
        x = xy[curr_vid, 0]
        y = xy[curr_vid, 1]
        ax.text(x, y, bid, size=7, c='1', ha='center', va='center')

    ax.set_title('Sensitivity wrt boundary pressures')

    ax.plot(xy[flow_network_sensitivity.boundary_vs, 0], xy[flow_network_sensitivity.boundary_vs, 1], 'o', color='r',
            markersize=10)

    cbar1 = plt.colorbar(line_segments)

    fig.savefig("output/hexa_distortion_std_2_sensitivity_neuman/sensitivity_eids"+str(i)+".png", dpi=200)

print(sensitivity_order)

# import matplotlib as mpl
# fig1, ax1 = plt.subplots(3,1,height_ratios=[2, 2, 1], figsize=(10,10))
#
# pos=ax1[0].matshow((inverse_model_sensitivity.sensitivity_matrix.todense().transpose()),cmap='RdBu',vmin=-.5,vmax=.5,aspect='auto')
# fig1.colorbar(pos,ax=ax1[0],location="top")
#
# pos=ax1[1].matshow(np.abs(inverse_model_sensitivity.sensitivity_matrix.todense().transpose()),vmin=0,vmax=.6,aspect='auto')
# fig1.colorbar(pos,ax=ax1[1],location="top")
#
# pos2=ax1[2].matshow(inverse_model_sensitivity.param_sensitivity.reshape(1,-1),vmin=0,vmax=.6,aspect='3')
# fig1.colorbar(pos2,ax=ax1[2],location="bottom")
#
# fig1.savefig("output/sensitivity_matrix.png", dpi=200)

# q_sensitivities = inverse_model_sensitivity.param_sensitivity

# # Plot network
# xy = flow_network_sensitivity.xyz[:, :2]
# edgelist = flow_network_sensitivity.edge_list
#
# segments = xy[edgelist]
#
# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# ax.set_xlim(xy[:, 0].min() - .5 * PARAMETERS['hexa_edge_length'], xy[:, 0].max() + PARAMETERS['hexa_edge_length'])
# ax.set_ylim(xy[:, 1].min() - .5 * PARAMETERS['hexa_edge_length'], xy[:, 1].max() + PARAMETERS['hexa_edge_length'])
# line_segments = LineCollection(segments, cmap=plt.get_cmap("viridis"),
#                                norm=plt.Normalize(vmin=0, vmax=np.max(q_sensitivities)))
# line_segments.set_array(q_sensitivities)
# line_segments.set_linewidth(5)
# ax.add_collection(line_segments)
#
# arg_sensitivity_sort = np.flip(np.argsort(inverse_model_sensitivity.param_sensitivity)[-8:])
#
# # for curr_eid, realisation in zip(arg_sensitivity_sort, range(8)):
# #     v_l = edgelist[curr_eid, 0]
# #     v_r = edgelist[curr_eid, 1]
# #     x = .5 * (xy[v_l, 0] + xy[v_r, 0])
# #     y = .5 * (xy[v_l, 1] + xy[v_r, 1])
# #
# #     ax.text(x, y, realisation + 1)
#
# for curr_eid in range(flow_network_sensitivity.nr_of_es):
#     v_l = edgelist[curr_eid, 0]
#     v_r = edgelist[curr_eid, 1]
#     x = .5 * (xy[v_l, 0] + xy[v_r, 0])
#     y = .5 * (xy[v_l, 1] + xy[v_r, 1])
#     ax.text(x, y, curr_eid,size=8,c='aqua', ha='center', va='center')
#
# for curr_vid, bid in zip(flow_network_sensitivity.boundary_vs,range(np.size(flow_network_sensitivity.boundary_vs))):
#     x = xy[curr_vid, 0]
#     y = xy[curr_vid, 1]
#     ax.text(x, y, bid, size=7, c='1', ha='center', va='center')
#
#
# ax.set_title('Sensitivity wrt boundary pressures')
#
# ax.plot(xy[flow_network_sensitivity.boundary_vs, 0], xy[flow_network_sensitivity.boundary_vs, 1], 'o', color='r', markersize=10)
#
# cbar1 = plt.colorbar(line_segments)
#
# fig.savefig("output/hexa_distortion_std_2_sensitivity_neuman/sensitivity_eids.png", dpi=200)

flowrate_ground_truth = copy.deepcopy(flow_network.flow_rate)
pressure_ground_truth = copy.deepcopy(flow_network.pressure)
boundary_pressure_ground_truth = copy.deepcopy(flow_network.boundary_val) - np.mean(flow_network.boundary_val)

norm_flowrate_to_truth_sens = {}
norm_flowrate_to_truth_neum = {}
norm_boundary_to_truth_sens = {}
norm_boundary_to_truth_neum = {}

nr_of_realisations = 81

for nr_of_meas in range(1, 9):

    flow_rates_opt_sensitivity = np.zeros((flow_network.nr_of_es, nr_of_realisations))
    flow_rates_opt_neuman = np.zeros((flow_network.nr_of_es, nr_of_realisations))

    boundary_pressure_opt_sensitivity = np.zeros((np.size(boundary_pressure_ground_truth), nr_of_realisations))
    boundary_pressure_opt_neuman = np.zeros((np.size(boundary_pressure_ground_truth), nr_of_realisations))

    for realisation in range(nr_of_realisations):

        # Initialise flownetwork and inverse model objects
        flow_network_sens_approach = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                                                 imp_solver, imp_velocity, PARAMETERS)
        flow_network_neum_approach = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                                                 imp_solver, imp_velocity, PARAMETERS)

        print("Read network: ...")
        flow_network_sens_approach.read_network()
        flow_network_neum_approach.read_network()
        print("Read network: DONE")

        print("Update transmissibility: ...")
        flow_network_sens_approach.update_transmissibility()
        flow_network_neum_approach.update_transmissibility()
        print("Update transmissibility: DONE")

        flow_network_sens_approach.transmiss = flow_network_sens_approach.transmiss * 0. + 1.
        flow_network_sens_approach.diameter = np.power(128. / np.pi * flow_network_sens_approach.transmiss * flow_network_sens_approach.length * flow_network_sens_approach.mu_rel * PARAMETERS["mu_plasma"], .25)
        flow_network_sens_approach.update_transmissibility()

        flow_network_neum_approach.transmiss = flow_network_neum_approach.transmiss * 0. + 1.
        flow_network_neum_approach.diameter = np.power(
            128. / np.pi * flow_network_neum_approach.transmiss * flow_network_neum_approach.length * flow_network_neum_approach.mu_rel *
            PARAMETERS["mu_plasma"], .25)
        flow_network_neum_approach.update_transmissibility()

        print("Update flow, pressure and velocity: ...")
        flow_network_sens_approach.update_blood_flow()
        flow_network_neum_approach.update_blood_flow()
        print("Update flow, pressure and velocity: DONE")

        # Distort
        inverse_model_sens_approach = InverseModel(flow_network_sens_approach, imp_readtargetvalues, imp_readparameters, imp_adjoint_parameter,
                                                   imp_adjoint_solver, imp_alpha_mapping, PARAMETERS)
        inverse_model_neum_approach = InverseModel(flow_network_neum_approach, imp_readtargetvalues, imp_readparameters, imp_adjoint_parameter,
                                                   imp_adjoint_solver, imp_alpha_mapping, PARAMETERS)

        nr_of_boundaries = inverse_model_sensitivity_base.nr_of_parameters

        distortion = np.random.normal(size=nr_of_boundaries, scale=2)  # scale is std of distortion
        flow_network_sens_approach.boundary_val += distortion
        flow_network_neum_approach.boundary_val += distortion

        # flow_network_sens_approach.boundary_val *= 0.
        # flow_network_rand_approach.boundary_val *= 0.

        print("Update flow, pressure and velocity: ...")
        flow_network_sens_approach.update_blood_flow()
        flow_network_neum_approach.update_blood_flow()
        print("Update flow, pressure and velocity: DONE")

        inverse_model_sens_approach.initialise_inverse_model()
        inverse_model_neum_approach.initialise_inverse_model()

        eid_meas_sens_approach = np.argpartition(inverse_model_sensitivity_base.param_sensitivity, -nr_of_meas)[-nr_of_meas:]
        eid_meas_neum_approach = sensitivity_order[:nr_of_meas]

        eid_max_sensitivity = np.argmax(inverse_model_sensitivity_base.param_sensitivity)

        inverse_model_sens_approach.edge_constraint_eid = np.array([eid_meas_sens_approach]).reshape(-1)
        inverse_model_sens_approach.nr_of_edge_constraints = np.size(inverse_model_sens_approach.edge_constraint_eid)
        inverse_model_sens_approach.edge_constraint_type = np.ones(inverse_model_sens_approach.nr_of_edge_constraints)
        inverse_model_sens_approach.edge_constraint_value = np.array([flowrate_ground_truth[eid_meas_sens_approach]]).reshape(-1)
        inverse_model_sens_approach.edge_constraint_range_pm = np.zeros(inverse_model_sens_approach.nr_of_edge_constraints)
        inverse_model_sens_approach.edge_constraint_sigma = np.ones(inverse_model_sens_approach.nr_of_edge_constraints) * np.abs(flowrate_ground_truth[eid_max_sensitivity])

        inverse_model_sens_approach.update_cost()

        print("Solve the inverse problem and update the boundaries: ...")
        j = 0
        while inverse_model_sens_approach.f_h > (1e-5 * nr_of_meas) and j < 30001:
            inverse_model_sens_approach.update_state()
            flow_network_sens_approach.update_transmissibility()
            flow_network_sens_approach.update_blood_flow()
            inverse_model_sens_approach.update_cost()
            if j % 5 == 0:
                print(str(j) + " iterations done (f_H =", "%.2e" % inverse_model_sens_approach.f_h + ")")
            j += 1
            if True in (flow_network_sens_approach.diameter < 0):
                import sys
                sys.exit("Negative diameter detected")

        print(str(j-1) + " iterations done (f_H =", "%.2e" % inverse_model_sens_approach.f_h + ")")
        print("Solve the inverse problem and update the boundaries: DONE")

        flow_rates_opt_sensitivity[:, realisation] = flow_network_sens_approach.flow_rate
        boundary_pressure_opt_sensitivity[:, realisation] = flow_network_sens_approach.boundary_val - np.mean(flow_network_sens_approach.boundary_val) #flow_network_sens_approach.boundary_val[0]

        # from numpy.random import default_rng
        # rng = default_rng()
        # numbers = rng.choice(flow_network.nr_of_es, size=nr_of_meas, replace=False)
        # eid_meas_rand_approach = numbers

        inverse_model_neum_approach.edge_constraint_eid = np.array([eid_meas_neum_approach]).reshape(-1)
        inverse_model_neum_approach.nr_of_edge_constraints = np.size(inverse_model_neum_approach.edge_constraint_eid)
        inverse_model_neum_approach.edge_constraint_type = np.ones(inverse_model_neum_approach.nr_of_edge_constraints)
        inverse_model_neum_approach.edge_constraint_value = np.array(
            [flowrate_ground_truth[eid_meas_neum_approach]]).reshape(-1)
        inverse_model_neum_approach.edge_constraint_range_pm = np.zeros(inverse_model_neum_approach.nr_of_edge_constraints)
        inverse_model_neum_approach.edge_constraint_sigma = np.ones(
            inverse_model_neum_approach.nr_of_edge_constraints) * np.abs(flowrate_ground_truth[eid_max_sensitivity])

        inverse_model_neum_approach.update_cost()

        print("Solve the inverse problem and update the boundaries: ...")
        j = 0
        while inverse_model_neum_approach.f_h > (1e-5 * nr_of_meas) and j < 30001:
            inverse_model_neum_approach.update_state()
            flow_network_neum_approach.update_transmissibility()
            flow_network_neum_approach.update_blood_flow()
            inverse_model_neum_approach.update_cost()
            if j % 5 == 0:
                print(str(j) + " iterations done (f_H =", "%.2e" % inverse_model_neum_approach.f_h + ")")
            j += 1
            if True in (flow_network_neum_approach.diameter < 0):
                import sys

                sys.exit("Negative diameter detected")

        print(str(j - 1) + " iterations done (f_H =", "%.2e" % inverse_model_neum_approach.f_h + ")")
        print("Solve the inverse problem and update the boundaries: DONE")

        flow_rates_opt_neuman[:, realisation] = flow_network_neum_approach.flow_rate
        boundary_pressure_opt_neuman[:, realisation] = flow_network_neum_approach.boundary_val - \
                                                       np.mean(flow_network_neum_approach.boundary_val)

    # Differences between tuned and ground truth flow rates
    diff_flowrate_to_truth_sens = flow_rates_opt_sensitivity - flowrate_ground_truth.reshape(-1, 1)
    diff_flowrate_to_truth_neum = flow_rates_opt_neuman - flowrate_ground_truth.reshape(-1, 1)

    # Differences between tuned and ground truth boundary pressures
    diff_bound_to_truth_sens = boundary_pressure_opt_sensitivity - boundary_pressure_ground_truth.reshape(-1, 1)
    diff_bound_to_truth_neum = boundary_pressure_opt_neuman - boundary_pressure_ground_truth.reshape(-1, 1)

    # L2 norm of all flow rate differences
    norm_flowrate_to_truth_sens[nr_of_meas] = np.linalg.norm(diff_flowrate_to_truth_sens,axis=0).tolist()
    norm_flowrate_to_truth_neum[nr_of_meas] = np.linalg.norm(diff_flowrate_to_truth_neum, axis=0).tolist()

    norm_boundary_to_truth_sens[nr_of_meas] = np.linalg.norm(diff_bound_to_truth_sens,axis=0).tolist()
    norm_boundary_to_truth_neum[nr_of_meas] = np.linalg.norm(diff_bound_to_truth_neum, axis=0).tolist()

import json

with open('output/hexa_distortion_std_2_sensitivity_neuman/norm_flowrate_to_truth_sens.json', 'w') as fp:
    json.dump(norm_flowrate_to_truth_sens, fp)

with open('output/hexa_distortion_std_2_sensitivity_neuman/norm_flowrate_to_truth_neum.json', 'w') as fp:
    json.dump(norm_flowrate_to_truth_neum, fp)

with open('output/hexa_distortion_std_2_sensitivity_neuman/norm_boundary_to_truth_sens.json', 'w') as fp:
    json.dump(norm_boundary_to_truth_sens, fp)

with open('output/hexa_distortion_std_2_sensitivity_neuman/norm_boundary_to_truth_neum.json', 'w') as fp:
    json.dump(norm_boundary_to_truth_neum, fp)





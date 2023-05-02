"""
A python script to simulate stationary blood flow in microvascular networks. Capabilities:
1. Import a network from file or generate a hexagonal network
2. Compute the edge transmissibilities with taking the impact of RBCs into account (Fahraeus, Fahraeus-Linquist effects)
3. Solve for flow rates, pressures and RBC velocities
4. Save the results in a file
"""

from source.flow_network import FlowNetwork
from source.inverse_model import InverseModel
from source_sensitivity.inverse_model_sensitivity import InverseModelSensitivity
from types import MappingProxyType
import source.setup.setup as setup
import copy
import json
import matplotlib.pyplot as plt


# MappingProxyType is basically a const dict.
# todo: read parameters from file; need a good way to import from human readable file (problem: json does not support
#  comments, which we would like to have in the text file; need better solution...)
PARAMETERS = dict( # MappingProxyType(
    {
        # Setup parameters for blood flow model
        "read_network_option": 2,  # 1: generate hexagonal graph
                                   # 2: import graph from csv files
                                   # 3: todo import graph from igraph files
                                   # 4: todo import graph from edge_data and vertex_data pickle files
        "write_network_option": 3,  # 1: do not write anything
                                    # 2: write to igraph format # todo: handle overwriting data from import file
                                    # 3: write to vtp format (paraview)
                                    # bla
        "tube_haematocrit_option": 2,  # 1: No RBCs (ht=0)
                                       # 2: Constant haematocrit
                                       # 3: todo: RBC tracking
                                       # 4-...: todo: steady state RBC laws
        "rbc_impact_option": 2,  # 1: No RBCs (hd=0)
                                 # 2: Laws by Pries, Neuhaus, Gaehtgens (1992)
                                 # 3-...: todo: Other laws. in vivo?
        "solver_option": 1,  # 1: Direct solver
                             # 2-...: other solvers (CG, AMG, ...)

        # Blood properties
        "ht_constant": 0.3,  # only required if RBC impact is considered
        "mu_plasma": 0.0012,

        # Hexagonal network properties. Only required for "read_network_option" 1
        "nr_of_hexagon_x": 3,
        "nr_of_hexagon_y": 3,
        "hexa_edge_length": 62.e-6,
        "hexa_diameter": 4.e-6,
        "hexa_boundary_vertices": [0, 27],
        "hexa_boundary_values": [2, 1],
        "hexa_boundary_types": [1, 1],

        # Import network from csv options. Only required for "read_network_option" 2
        "csv_path_vertex_data": "data/network/mvn_1_edit_full/node_data.csv",
        "csv_path_edge_data": "data/network/mvn_1_edit_full/edge_data.csv",
        "csv_path_boundary_data": "data/network/mvn_1_edit_full/node_boundary_data.csv",
        "csv_diameter": "D", "csv_length": "L",
        "csv_edgelist_v1": "n1", "csv_edgelist_v2": "n2",
        "csv_coord_x": "x", "csv_coord_y": "y", "csv_coord_z": "z",
        "csv_boundary_vs": "nodeId", "csv_boundary_type": "boundaryType", "csv_boundary_value": "p",

        # Write options
        "write_override_initial_graph": False,  # todo: currently does not do anything
        "write_path_igraph": "data/network/small.vtp",  # only required for "write_network_option" 2, 3

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
        "csv_path_edge_target_data": "data/inverse_model/edge_target.csv",
        # Filepath to define the edge parameter space (only for tuning of diameters and transmissibilities)
        "csv_path_edge_parameterspace": "not needed",
        # Filepath to define the vertex parameter space (only for tuning of boundary conditions)
        "csv_path_vertex_parameterspace": "data/inverse_model/vertex_parameters.csv",
        # Gradient descent options:
        "gamma": 500,
        "phi": .5,  # for parameter_restriction 2
        "max_nr_of_iterations": 50  # Maximum of iterations
    }
)

# Create object to set up the simulation and initialise the simulation
setup_simulation = setup.SetupSimulation()
# Initialise the implementations based on the parameters specified
imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_velocity, imp_buildsystem, \
    imp_solver = setup_simulation.setup_bloodflow_model(PARAMETERS)

# Build flownetwork object and pass the implementations of the different submodules, which were selected in
#  the parameter file
flow_network_mvn1 = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                           imp_solver, imp_velocity, PARAMETERS)

# Import or generate the network
print("Read network: ...")
flow_network_mvn1.read_network()
print("Read network: DONE")

# Update the transmissibility
print("Update transmissibility: ...")
flow_network_mvn1.update_transmissibility()
print("Update transmissibility: DONE")

# Update flow rate, pressure and RBC velocity
print("Update flow, pressure and velocity: ...")
flow_network_mvn1.update_blood_flow()
print("Update flow, pressure and velocity: DONE")

# Write the results to file
flow_network_mvn1.write_network()

import igraph
import numpy as np

edge_list = flow_network_mvn1.edge_list
graph = igraph.Graph(edge_list.tolist())  # Generate igraph based on edge_list

graph.es["diameter"] = flow_network_mvn1.diameter
graph.es["length"] = flow_network_mvn1.length
graph.es["flow_rate"] = flow_network_mvn1.flow_rate
graph.es["rbc_velocity"] = flow_network_mvn1.rbc_velocity
graph.vs["xyz"] = flow_network_mvn1.xyz.tolist()
graph.vs["pressure"] = flow_network_mvn1.pressure

graph.vs["degree"] = graph.degree()

xyz_min = np.min(flow_network_mvn1.xyz, axis=0)
xyz_max = np.max(flow_network_mvn1.xyz, axis=0)

size_box = 200.e-6  # in um

cut_min = .5 * (xyz_max + xyz_min) - size_box/2.
cut_max = .5 * (xyz_max + xyz_min) + size_box/2.

is_vs_in_cut_xyz = np.logical_and(flow_network_mvn1.xyz > cut_min, flow_network_mvn1.xyz < cut_max)
is_vs_in_cut = np.all(is_vs_in_cut_xyz, axis=1)

vs_ids_in_cut = np.arange(graph.vcount())[is_vs_in_cut]
vs_ids_neighbourhood = np.unique(np.hstack(graph.neighborhood(vs_ids_in_cut)))

g_subgraph = graph.induced_subgraph(vs_ids_neighbourhood)

print("Total graph:")
print(g_subgraph.summary())

g_subgraph_decompose = g_subgraph.decompose()
n_vs_decomposed = [g_subgraph_decompose[i].vcount() for i in range(len(g_subgraph_decompose))]
g_subgraph_largest = g_subgraph_decompose[np.argmax(n_vs_decomposed)]  # Decompose subgraph in largest connected comp
print("Largest subgraph:")
print(g_subgraph_largest.summary())

# Reference network
flow_network_small_ref = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                                     imp_solver, imp_velocity, PARAMETERS)

flow_network_small_ref.nr_of_vs = g_subgraph_largest.vcount()
flow_network_small_ref.nr_of_es = g_subgraph_largest.ecount()

# Edge attributes
flow_network_small_ref.length = np.array(g_subgraph_largest.es["length"])
flow_network_small_ref.diameter = np.array(g_subgraph_largest.es["diameter"])
flow_network_small_ref.flow_rate = g_subgraph_largest.es["flow_rate"]
flow_network_small_ref.rbc_velocity = g_subgraph_largest.es["rbc_velocity"]
flow_network_small_ref.edge_list = np.array(g_subgraph_largest.get_edgelist())

# Vertex attributes
flow_network_small_ref.xyz = np.array(g_subgraph_largest.vs["xyz"])
flow_network_small_ref.pressure = np.array(g_subgraph_largest.vs["pressure"])

is_boundary = np.array(g_subgraph_largest.degree()) != np.array(g_subgraph_largest.vs["degree"])

boundary_vs_id = np.arange(g_subgraph_largest.vcount())[is_boundary]
boundary_value = flow_network_small_ref.pressure[is_boundary]
boundary_type = np.ones(np.size(boundary_vs_id))

PARAMETERS["write_path_igraph"] = "output/realistic_box_200_sensitivity_edgeremove/small_ref.vtp"
flow_network_small_ref.write_network()

print("Nr of es: ", g_subgraph_largest.ecount())
print("Nr of vs: ", g_subgraph_largest.vcount())
print("Nr of bs: ", np.size(boundary_vs_id))

flow_network_small_ref.boundary_vs = boundary_vs_id
flow_network_small_ref.boundary_val = boundary_value
flow_network_small_ref.boundary_type = boundary_type

print("Update transmissibility: ...")
flow_network_small_ref.update_transmissibility()
print("Update transmissibility: DONE")

# Update flow rate, pressure and RBC velocity
print("Update flow, pressure and velocity: ...")
flow_network_small_ref.update_blood_flow()
print("Update flow, pressure and velocity: DONE")

# Network for sensitivity calculation
flow_network_small_sensitivity = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                                     imp_solver, imp_velocity, PARAMETERS)

flow_network_small_sensitivity.nr_of_vs = g_subgraph_largest.vcount()
flow_network_small_sensitivity.nr_of_es = g_subgraph_largest.ecount()

# Edge attributes
flow_network_small_sensitivity.length = np.array(g_subgraph_largest.es["length"])
flow_network_small_sensitivity.diameter = np.array(g_subgraph_largest.es["diameter"])
flow_network_small_sensitivity.flow_rate = g_subgraph_largest.es["flow_rate"]
flow_network_small_sensitivity.rbc_velocity = g_subgraph_largest.es["rbc_velocity"]
flow_network_small_sensitivity.edge_list = np.array(g_subgraph_largest.get_edgelist())

# Vertex attributes
flow_network_small_sensitivity.xyz = np.array(g_subgraph_largest.vs["xyz"])
flow_network_small_sensitivity.pressure = np.array(g_subgraph_largest.vs["pressure"])

flow_network_small_sensitivity.boundary_vs = boundary_vs_id
flow_network_small_sensitivity.boundary_val = boundary_value
flow_network_small_sensitivity.boundary_type = boundary_type

print("Update transmissibility: ...")
flow_network_small_sensitivity.update_transmissibility()
print("Update transmissibility: DONE")

# Update flow rate, pressure and RBC velocity
print("Update flow, pressure and velocity: ...")
flow_network_small_sensitivity.update_blood_flow()
print("Update flow, pressure and velocity: DONE")

# Start with finding sensitivity order
global_eids = np.arange(flow_network_small_ref.nr_of_es)
sensitivity_order = np.array([], dtype=int)

# Initialise objects related to the inverse model.
imp_readtargetvalues, imp_readparameters, imp_adjoint_parameter, imp_adjoint_solver, imp_alpha_mapping = setup_simulation.setup_inverse_model(PARAMETERS)

inverse_model_sensitivity_base = InverseModelSensitivity(flow_network_small_ref, imp_readtargetvalues, imp_readparameters,
                                                         imp_adjoint_parameter, imp_adjoint_solver, imp_alpha_mapping, PARAMETERS)

inverse_model_sensitivity_base.initialise_inverse_model()
inverse_model_sensitivity_base.update_sensitivity()

boundary = np.zeros(flow_network_small_ref.nr_of_vs)
boundary[is_boundary] = 1

flow_network_small_ref.wildcard_vertex_attr = boundary
flow_network_small_ref.wildcard_edge_attr = inverse_model_sensitivity_base.param_sensitivity

flow_network_small_ref.write_network()

flowrate_ground_truth = copy.deepcopy(flow_network_small_ref.flow_rate)
pressure_ground_truth = copy.deepcopy(flow_network_small_ref.pressure)
boundary_pressure_ground_truth = copy.deepcopy(flow_network_small_ref.boundary_val) - np.mean(flow_network_small_ref.boundary_val)

norm_flowrate_to_truth_sens = {}
norm_flowrate_to_truth_rand = {}
norm_boundary_to_truth_sens = {}
norm_boundary_to_truth_rand = {}

inverse_model_sensitivity_neum = InverseModelSensitivity(flow_network_small_sensitivity, imp_readtargetvalues, imp_readparameters,
                                                         imp_adjoint_parameter, imp_adjoint_solver, imp_alpha_mapping, PARAMETERS)

inverse_model_sensitivity_neum.initialise_inverse_model()
inverse_model_sensitivity_neum.update_sensitivity()

edge_id_remove_local = -1

for i in range(52):
    if np.size(sensitivity_order) > 0 and edge_id_remove_local >=0:

        current_remove = edge_id_remove_local

        flow_network_small_sensitivity.edge_list = np.delete(flow_network_small_sensitivity.edge_list, current_remove, axis=0)
        flow_network_small_sensitivity.diameter = np.delete(flow_network_small_sensitivity.diameter, current_remove)
        flow_network_small_sensitivity.length = np.delete(flow_network_small_sensitivity.length, current_remove)
        flow_network_small_sensitivity.ht = np.delete(flow_network_small_sensitivity.ht, current_remove)
        flow_network_small_sensitivity.hd = np.delete(flow_network_small_sensitivity.hd, current_remove)
        flow_network_small_sensitivity.mu_rel = np.delete(flow_network_small_sensitivity.mu_rel, current_remove)
        flow_network_small_sensitivity.transmiss = np.delete(flow_network_small_sensitivity.transmiss, current_remove)
        flow_network_small_sensitivity.flow_rate = np.delete(flow_network_small_sensitivity.flow_rate, current_remove)
        flow_network_small_sensitivity.rbc_velocity = np.delete(flow_network_small_sensitivity.rbc_velocity, current_remove)
        flow_network_small_sensitivity.nr_of_es = flow_network_small_sensitivity.nr_of_es - 1
        global_eids = np.delete(global_eids, current_remove)

        flow_network_small_sensitivity.update_linear_system()

        inverse_model_sensitivity_neum = InverseModelSensitivity(flow_network_small_sensitivity, imp_readtargetvalues,
                                                            imp_readparameters, imp_adjoint_parameter,
                                                            imp_adjoint_solver, imp_alpha_mapping, PARAMETERS)

        inverse_model_sensitivity_neum.initialise_inverse_model()
        inverse_model_sensitivity_neum.update_sensitivity()

    edge_id_remove_local = np.argmax(inverse_model_sensitivity_neum.param_sensitivity)
    edge_id_remove_global = global_eids[edge_id_remove_local]
    sensitivity_order = np.append(sensitivity_order, edge_id_remove_global)

nr_of_realisations = 1

# nr_of_realisations = 1

for nr_of_meas in range(1, 29):
#for nr_of_meas in range(1, 9):

    flow_rates_opt_sensitivity = np.zeros((flow_network_small_ref.nr_of_es, nr_of_realisations))

    boundary_pressure_opt_sensitivity = np.zeros((np.size(boundary_pressure_ground_truth), nr_of_realisations))

    eid_meas_sens_approach = np.array([])
    eid_meas_neum_approach = np.array([])

    for realisation in range(nr_of_realisations):
        print("#################################")
        print("nr of meas:", nr_of_meas, ", realisation:", realisation)
        print("#################################")

        # Initialise flownetwork and inverse model objects
        flow_network_sens_approach = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                                                 imp_solver, imp_velocity, PARAMETERS)

        # use above network
        print("Read network: ...")

        # Basically copy the reference
        flow_network_sens_approach.nr_of_vs = np.copy(flow_network_small_ref.nr_of_vs)
        flow_network_sens_approach.nr_of_es = np.copy(flow_network_small_ref.nr_of_es)
        flow_network_sens_approach.length = np.copy(flow_network_small_ref.length)
        flow_network_sens_approach.diameter = np.copy(flow_network_small_ref.diameter)
        flow_network_sens_approach.flow_rate = np.copy(flow_network_small_ref.flow_rate)
        flow_network_sens_approach.rbc_velocity = np.copy(flow_network_small_ref.rbc_velocity)
        flow_network_sens_approach.edge_list = np.copy(flow_network_small_ref.edge_list)
        flow_network_sens_approach.xyz = np.copy(flow_network_small_ref.xyz)
        flow_network_sens_approach.pressure = np.copy(flow_network_small_ref.pressure)
        flow_network_sens_approach.boundary_vs = np.copy(flow_network_small_ref.boundary_vs)
        flow_network_sens_approach.boundary_val = np.copy(flow_network_small_ref.boundary_val)
        flow_network_sens_approach.boundary_type = np.copy(flow_network_small_ref.boundary_type)
        print("Read network: DONE")

        print("Update transmissibility: ...")
        flow_network_sens_approach.update_transmissibility()
        print("Update transmissibility: DONE")

        print("Update flow, pressure and velocity: ...")
        flow_network_sens_approach.update_blood_flow()
        print("Update flow, pressure and velocity: DONE")

        # Distort
        inverse_model_sens_approach = InverseModel(flow_network_sens_approach, imp_readtargetvalues, imp_readparameters, imp_adjoint_parameter,
                                                   imp_adjoint_solver, imp_alpha_mapping, PARAMETERS)

        nr_of_boundaries = inverse_model_sensitivity_base.nr_of_parameters

        distortion = np.random.normal(size=nr_of_boundaries, scale=100)  # scale is std of distortion

        flow_network_sens_approach.boundary_val *= 0.
        # flow_network_sens_approach.boundary_val += distortion

        print("Update flow, pressure and velocity: ...")
        flow_network_sens_approach.update_blood_flow()
        print("Update flow, pressure and velocity: DONE")

        inverse_model_sens_approach.initialise_inverse_model()

        #eid_meas_sens_approach = np.argpartition(inverse_model_sensitivity_base.param_sensitivity, -nr_of_meas)[-nr_of_meas:]
        eid_meas_neum_approach = sensitivity_order[:nr_of_meas]

        eid_max_sensitivity = np.argmax(inverse_model_sensitivity_base.param_sensitivity)

        inverse_model_sens_approach.edge_constraint_eid = np.array([eid_meas_neum_approach]).reshape(-1)
        inverse_model_sens_approach.nr_of_edge_constraints = np.size(inverse_model_sens_approach.edge_constraint_eid)
        inverse_model_sens_approach.edge_constraint_type = np.ones(inverse_model_sens_approach.nr_of_edge_constraints)
        inverse_model_sens_approach.edge_constraint_value = np.array([flowrate_ground_truth[eid_meas_neum_approach]]).reshape(-1)
        inverse_model_sens_approach.edge_constraint_range_pm = np.zeros(inverse_model_sens_approach.nr_of_edge_constraints)
        inverse_model_sens_approach.edge_constraint_sigma = np.ones(inverse_model_sens_approach.nr_of_edge_constraints) * np.abs(flowrate_ground_truth[eid_max_sensitivity])

        inverse_model_sens_approach.update_cost()

        print("Solve the inverse problem and update the boundaries (Sensitivity approach): ...")
        j = 0
        while inverse_model_sens_approach.f_h > (1e-5 * nr_of_meas) and j < 500001:
            inverse_model_sens_approach.update_state()
            flow_network_sens_approach.update_transmissibility()
            flow_network_sens_approach.update_blood_flow()
            inverse_model_sens_approach.update_cost()
            if j % 200 == 0:
                print(str(j) + " iterations done (f_H =", "%.2e" % inverse_model_sens_approach.f_h + ")")
            j += 1

        print(str(j-1) + " iterations done (f_H =", "%.2e" % inverse_model_sens_approach.f_h + ")")
        print("Solve the inverse problem and update the boundaries (Sensitivity approach): DONE")

        PARAMETERS["write_path_igraph"] = "output/realistic_box_200_sensitivity_edgeremove/small_opt_meas_"+str(nr_of_meas)+"_real_"+str(realisation)+".vtp"

        nr_of_flowrates_vtp = np.size(flowrate_ground_truth)
        is_target_vtp = np.zeros(nr_of_flowrates_vtp)
        is_target_vtp[eid_meas_neum_approach] = 1

        flow_network_sens_approach.wildcard_edge_attr = np.copy(is_target_vtp)
        flow_network_sens_approach.wildcard_vertex_attr = np.copy(boundary)

        flow_network_sens_approach.write_network()

        flow_rates_opt_sensitivity[:, realisation] = flow_network_sens_approach.flow_rate
        boundary_pressure_opt_sensitivity[:, realisation] = flow_network_sens_approach.boundary_val - np.mean(flow_network_sens_approach.boundary_val) #flow_network_sens_approach.boundary_val[0]


    # Differences between tuned and ground truth flow rates
    diff_flowrate_to_truth_sens = flow_rates_opt_sensitivity - flowrate_ground_truth.reshape(-1, 1)

    # Differences between tuned and ground truth boundary pressures
    diff_bound_to_truth_sens = boundary_pressure_opt_sensitivity - boundary_pressure_ground_truth.reshape(-1, 1)

    # L2 norm of all flow rate differences
    norm_flowrate_to_truth_sens[nr_of_meas] = np.linalg.norm(diff_flowrate_to_truth_sens,axis=0).tolist()

    current_norm = norm_flowrate_to_truth_sens[nr_of_meas][0]

    norm_boundary_to_truth_sens[nr_of_meas] = np.linalg.norm(diff_bound_to_truth_sens,axis=0).tolist()

    nr_of_flowrates = np.size(flowrate_ground_truth)

    is_target = np.zeros(nr_of_flowrates)
    is_target[eid_meas_neum_approach] = 1

    # size = (is_target*4 + 1.) * 6
    #
    # is_dir_right = np.sign(flowrate_ground_truth) * np.sign(flow_rates_opt_sensitivity.reshape(-1))
    # is_dir_right[is_dir_right<0] = 0
    #
    # color = np.copy(is_dir_right)
    # is_dir_right[eid_meas_neum_approach] = 2
    #
    # rel_error = np.abs((flowrate_ground_truth - flow_rates_opt_sensitivity.reshape(-1)) / flowrate_ground_truth)
    # rel_error_median = np.median(rel_error)
    #
    # # fig1, [ax1, ax2] = plt.subplots(1, 2, figsize=(8,4))
    # fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    # ax1.scatter(np.abs(flowrate_ground_truth), np.abs(flow_rates_opt_sensitivity), s=size, c = is_dir_right, cmap = "rainbow", zorder=2)
    # min_flow_val = 1.e-20  # np.min(np.append(np.abs(flowrate_ground_truth), np.abs(flow_rates_opt_sensitivity)))
    # max_flow_val = 2.e-13  # np.max(np.append(np.abs(flowrate_ground_truth), np.abs(flow_rates_opt_sensitivity)))
    # ax1.plot([min_flow_val, max_flow_val], [min_flow_val, max_flow_val], 'k-', zorder=1)
    # ax1.set_xscale("log")
    # ax1.set_yscale("log")
    # ax1.set_title('Nr. of meas = '+str(nr_of_meas)+', L2 = '+ '%.2E' % current_norm+', median error = '+'%.2E' % rel_error_median, size =8)
    #
    # ax1.set_xlabel('Flow rates ground truth')
    # ax1.set_ylabel('Flow rates optimised')
    #
    # ax1.set_xlim([min_flow_val, max_flow_val])
    # ax1.set_ylim([min_flow_val, max_flow_val])
    #
    # # rel_difference = (flow_rates_opt_sensitivity.reshape(-1) - flowrate_ground_truth) / flowrate_ground_truth
    #
    # # ax2.scatter(np.abs(flowrate_ground_truth), rel_difference)
    #
    # plt.tight_layout()
    #
    # fig1.savefig("output/realistic_box_200_sensitivity_edgeremove/flowrate_scatter_nr_meas_"+str(nr_of_meas)+".png", dpi=200)

    with open('output/realistic_box_200_sensitivity_edgeremove/norm_flowrate_to_truth_sens_remove.json', 'w') as fp:
        json.dump(norm_flowrate_to_truth_sens, fp)

    with open('output/realistic_box_200_sensitivity_edgeremove/norm_boundary_to_truth_sens_remove.json', 'w') as fp:
        json.dump(norm_boundary_to_truth_sens, fp)

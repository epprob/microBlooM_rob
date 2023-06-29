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
        # Blood properties (alles unwichtig, da hier die transmissibility auf 1 enforced wird)
        "ht_constant": 0.0,
        "mu_plasma": 0.0012,
        # Hexagonal network properties (Anzahl hexagons in x und y-Richtung)
        "nr_of_hexagon_x": 5,
        "nr_of_hexagon_y": 5,
        "hexa_edge_length": 62.e-6,  # Ist hier nur relevant für die Darstellung, da Transmissibility fix ist
        "hexa_diameter": 4.e-6,  # Ist hier nur relevant für die Darstellung, da Transmissibility fix ist
        "hexa_boundary_vertices": [0, 65, 21, 48, 41, 31, 56],  # Vertex ids, die eine boundary sind
        "hexa_boundary_values": [2, 1, 3, 1, 1, 5, 2],  # Ground-truth werte der boundary pressures
        "hexa_boundary_types": [1, 1, 1, 1, 1, 1, 1],  # 1: pressure, 2: flow rate
        # Import network from csv options
        "csv_path_vertex_data": "not needed here",
        "csv_path_edge_data": "not needed here",
        "csv_path_boundary_data": "not needed here",
        "csv_diameter": "not needed here", "csv_length": "not needed here",
        "csv_edgelist_v1": "not needed here", "csv_edgelist_v2": "not needed here",
        "csv_coord_x": "not needed here", "csv_coord_y": "not needed here", "csv_coord_z": "not needed here",
        "csv_boundary_vs": "not needed here", "csv_boundary_type": "not needed here",
        "csv_boundary_value": "not needed here",
        # Write options
        "write_override_initial_graph": False,
        "write_path_igraph": "not needed here", # only required for "write_network_option" 2
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
        # Filepath to prescribe target values / constraints on edges (eigentlich braucht es dieses file hier nicht,
        # da die target values ja anhand der Sensitivitäten ausgewählt werden. Habe diesen Fall aber noch nicht
        # sauber implementiert, daher muss man ein file vorgeben (sorry)
        "csv_path_edge_target_data": "data/inverse_model/edge_target_not_needed.csv",
        # Filepath to define the edge parameter space (only for tuning of diameters and transmissibilities)
        "csv_path_edge_parameterspace": "not needed",
        # Filepath to define the vertex parameter space (only for tuning of boundary conditions)
        # all heisst, dass die drücke in allen boundary vertices getuned werden
        "csv_path_vertex_parameterspace": "data/inverse_model/vertex_parameters.csv",
        # Gradient descent options:
        "gamma": .05,
        "phi": .5,  # for parameter_restriction 2 (slope of tanh)
        "max_nr_of_iterations": 50  # Maximum of iterations
    }
)

# Construct SetupSimulation object
setup_simulation = setup.SetupSimulation()

# Initialise objects related to simulate blood flow without RBC tracking.
imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_velocity, imp_buildsystem, \
    imp_solver = setup_simulation.setup_bloodflow_model(PARAMETERS)

# Initialise objects related to the inverse model.
imp_readtargetvalues, imp_readparameters, imp_adjoint_parameter, imp_adjoint_solver, \
    imp_alpha_mapping = setup_simulation.setup_inverse_model(PARAMETERS)

# Construct flow network object for the ground truth network
flow_network_ground_truth = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                                        imp_solver, imp_velocity, PARAMETERS)
# Construct inverse_model_sensitivity object to compute the sensitivity of all edges
inverse_model_sensitivity = InverseModelSensitivity(flow_network_ground_truth, imp_readtargetvalues, imp_readparameters, imp_adjoint_parameter,
                                                    imp_adjoint_solver, imp_alpha_mapping, PARAMETERS)

print("Read network: ...")
flow_network_ground_truth.read_network()  # Generate the hexagonal network
print("Read network: DONE")

print("Update transmissibility: ...")
flow_network_ground_truth.update_transmissibility()  # Compute the transmissibilities of all edges based on the given network
print("Update transmissibility: DONE")

# Achtung Hack: Hier enforce ich alle transmissibilities auf 1. Ausserdem setze ich die Durchmesser so, dass
# sie mit der transmissibility=1 konsistent sind.
flow_network_ground_truth.transmiss = flow_network_ground_truth.transmiss * 0. + 1.
flow_network_ground_truth.diameter = np.power(128. / np.pi * flow_network_ground_truth.transmiss * flow_network_ground_truth.length * flow_network_ground_truth.mu_rel * PARAMETERS["mu_plasma"], .25)

flow_network_ground_truth.update_transmissibility()  # Sollte nun 1 sein überall

print("Update flow, pressure and velocity: ...")
flow_network_ground_truth.update_blood_flow()  # Baut das lineare System basierend auf dem Netzwerk, berechnet druck und flow
print("Update flow, pressure and velocity: DONE")

# Initialisiert das inverse Modell (liest, welche boundary vertices parameter sind, und somit getuned werden)
inverse_model_sensitivity.initialise_inverse_model()

# Berechnet die Sensitivitäts matrix (inverse_model_sensitivity.sensitivity_matrix) und die L2 norm davon
# (inverse_model_sensitivity.param_sensitivity)
inverse_model_sensitivity.update_sensitivity()

# Erstellt ein plot der Sensitivitäten
import matplotlib as mpl
fig1, ax1 = plt.subplots(3,1,height_ratios=[2, 2, 1], figsize=(10,10))

pos=ax1[0].matshow((inverse_model_sensitivity.sensitivity_matrix.todense().transpose()),cmap='RdBu',vmin=-.5,vmax=.5,aspect='auto')
fig1.colorbar(pos,ax=ax1[0],location="top")

pos=ax1[1].matshow(np.abs(inverse_model_sensitivity.sensitivity_matrix.todense().transpose()),vmin=0,vmax=.6,aspect='auto')
fig1.colorbar(pos,ax=ax1[1],location="top")

pos2=ax1[2].matshow(inverse_model_sensitivity.param_sensitivity.reshape(1,-1),vmin=0,vmax=.6,aspect='3')
fig1.colorbar(pos2,ax=ax1[2],location="bottom")

fig1.savefig("output/sensitivity_matrix.png", dpi=200)

q_sensitivities = inverse_model_sensitivity.param_sensitivity

# Plot network
xy = flow_network_ground_truth.xyz[:, :2]
edgelist = flow_network_ground_truth.edge_list

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

for curr_eid in range(flow_network_ground_truth.nr_of_es):
    v_l = edgelist[curr_eid, 0]
    v_r = edgelist[curr_eid, 1]
    x = .5 * (xy[v_l, 0] + xy[v_r, 0])
    y = .5 * (xy[v_l, 1] + xy[v_r, 1])
    ax.text(x, y, curr_eid,size=8,c='aqua', ha='center', va='center')

for curr_vid, bid in zip(flow_network_ground_truth.boundary_vs, range(np.size(flow_network_ground_truth.boundary_vs))):
    x = xy[curr_vid, 0]
    y = xy[curr_vid, 1]
    ax.text(x, y, bid, size=7, c='1', ha='center', va='center')


ax.set_title('Sensitivity wrt boundary pressures')

ax.plot(xy[flow_network_ground_truth.boundary_vs, 0], xy[flow_network_ground_truth.boundary_vs, 1], 'o', color='r', markersize=10)

cbar1 = plt.colorbar(line_segments)

# plt.show()

fig.savefig("output/sensitivity_eids.png", dpi=200)

# Kopiere die Ground truth werte für später
flowrate_ground_truth = copy.deepcopy(flow_network_ground_truth.flow_rate)
pressure_ground_truth = copy.deepcopy(flow_network_ground_truth.pressure)
boundary_pressure_ground_truth = copy.deepcopy(flow_network_ground_truth.boundary_val) - np.mean(flow_network_ground_truth.boundary_val)

# Tuning of boundary conditions based on measurements.

norm_flowrate_to_truth_sens = {}  # Error norm flow rates after tuning compared to ground truth, if measurement locations are chosen with sensitivity approach
norm_flowrate_to_truth_rand = {}  # Error norm flow rates after tuning compared to ground truth, if measurement locations are chosen randomly
norm_boundary_to_truth_sens = {}  # Error norm boundary pressures after tuning compared to ground truth, if measurement locations are chosen with sensitivity approach
norm_boundary_to_truth_rand = {}  # Error norm boundary pressures after tuning compared to ground truth, if measurement locations are chosen randomly

nr_of_realisations = 3 # 51 # Anzahl samples verschiedener prior Boundary-drücke

for nr_of_meas in range(1, 9):  # Variere die Anzahl Messungen von 1 bis 9

    # Flow rates nach tuning in allen edges für alle realisierungen
    flow_rates_opt_sensitivity = np.zeros((flow_network_ground_truth.nr_of_es, nr_of_realisations))
    flow_rates_opt_random = np.zeros((flow_network_ground_truth.nr_of_es, nr_of_realisations))

    # Boundary pressures nach tuning in allen edges für alle realisierungen
    boundary_pressure_opt_sensitivity = np.zeros((np.size(boundary_pressure_ground_truth), nr_of_realisations))
    boundary_pressure_opt_random = np.zeros((np.size(boundary_pressure_ground_truth), nr_of_realisations))

    for realisation in range(nr_of_realisations):

        # Initialise flownetwork and inverse model objects (für aktuelle realisierung)
        flow_network_sens_approach = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                                                 imp_solver, imp_velocity, PARAMETERS)
        flow_network_rand_approach = FlowNetwork(imp_readnetwork, imp_writenetwork, imp_ht, imp_hd, imp_transmiss, imp_buildsystem,
                                                 imp_solver, imp_velocity, PARAMETERS)

        print("Read network: ...")
        flow_network_sens_approach.read_network()  # Sensitivitäts Ansatz
        flow_network_rand_approach.read_network()  # Random Wahl Ansatz
        print("Read network: DONE")

        print("Update transmissibility: ...")
        flow_network_sens_approach.update_transmissibility()
        flow_network_rand_approach.update_transmissibility()
        print("Update transmissibility: DONE")

        flow_network_sens_approach.transmiss = flow_network_sens_approach.transmiss * 0. + 1.
        flow_network_sens_approach.diameter = np.power(128. / np.pi * flow_network_sens_approach.transmiss * flow_network_sens_approach.length * flow_network_sens_approach.mu_rel * PARAMETERS["mu_plasma"], .25)
        flow_network_sens_approach.update_transmissibility()

        flow_network_rand_approach.transmiss = flow_network_rand_approach.transmiss * 0. + 1.
        flow_network_rand_approach.diameter = np.power(
            128. / np.pi * flow_network_rand_approach.transmiss * flow_network_rand_approach.length * flow_network_rand_approach.mu_rel *
            PARAMETERS["mu_plasma"], .25)
        flow_network_rand_approach.update_transmissibility()

        print("Update flow, pressure and velocity: ...")
        flow_network_sens_approach.update_blood_flow()
        flow_network_rand_approach.update_blood_flow()
        print("Update flow, pressure and velocity: DONE")

        # Inverse Modell Opjekte für Sensitivität und Random Ansatz
        inverse_model_sens_approach = InverseModel(flow_network_sens_approach, imp_readtargetvalues, imp_readparameters, imp_adjoint_parameter,
                                                   imp_adjoint_solver, imp_alpha_mapping, PARAMETERS)
        inverse_model_rand_approach = InverseModel(flow_network_rand_approach, imp_readtargetvalues, imp_readparameters, imp_adjoint_parameter,
                                                   imp_adjoint_solver, imp_alpha_mapping, PARAMETERS)

        nr_of_boundaries = inverse_model_sensitivity.nr_of_parameters

        # Distortion der boundary pressures von der Ground-truth, um einen Prior für das Sample zu erzeugen
        distortion = np.random.normal(size=nr_of_boundaries, scale=2)  # scale is std of distortion

        flow_network_sens_approach.boundary_val += distortion
        flow_network_rand_approach.boundary_val += distortion

        print("Update flow, pressure and velocity: ...")
        flow_network_sens_approach.update_blood_flow()
        flow_network_rand_approach.update_blood_flow()
        print("Update flow, pressure and velocity: DONE")

        inverse_model_sens_approach.initialise_inverse_model()
        inverse_model_rand_approach.initialise_inverse_model()

        # Wähle die "nr_of_meas" edge ids aus mit den "nr_of_meas" höchsten Sensitivitäten.
        eid_meas_sens_approach = np.argpartition(inverse_model_sensitivity.param_sensitivity, -nr_of_meas)[-nr_of_meas:]
        # Edge id mit höchster Sensitivität
        eid_max_sensitivity = np.argmax(inverse_model_sensitivity.param_sensitivity)

        # Setze die Messungen für das Inverse Modell basierend auf der Ground truth
        inverse_model_sens_approach.edge_constraint_eid = np.array([eid_meas_sens_approach]).reshape(-1)  # Array mit allen edge ids, die eine Messung sind
        inverse_model_sens_approach.nr_of_edge_constraints = np.size(inverse_model_sens_approach.edge_constraint_eid)  # Anzahl Messungen
        inverse_model_sens_approach.edge_constraint_type = np.ones(inverse_model_sens_approach.nr_of_edge_constraints)  # Typ 1 bedeutet flow rate messwert
        inverse_model_sens_approach.edge_constraint_value = np.array([flowrate_ground_truth[eid_meas_sens_approach]]).reshape(-1)  # Messwerte sind die flow rates der Ground truth
        inverse_model_sens_approach.edge_constraint_range_pm = np.zeros(inverse_model_sens_approach.nr_of_edge_constraints)  # 0 bedeutet, dass der Messwert keine Toleranz hat (Ansonsten könnte ein plus/minus Wert angegeben werden)
        inverse_model_sens_approach.edge_constraint_sigma = np.ones(inverse_model_sens_approach.nr_of_edge_constraints) * np.abs(flowrate_ground_truth[eid_max_sensitivity])  # Sigma normalisiert die Cost function Werte (f=sum((q_i-q_i^tar)/sigma_i)

        inverse_model_sens_approach.update_cost()  # Berechnet den cost function wert

        print("Solve the inverse problem and update the boundaries: ...")
        j = 0
        # Löse das Inverse Problem mit dem Sensitivitäts Ansatz
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

        # Speichere die optimierten Flow rates und boundaries ab
        flow_rates_opt_sensitivity[:, realisation] = flow_network_sens_approach.flow_rate
        boundary_pressure_opt_sensitivity[:, realisation] = flow_network_sens_approach.boundary_val - np.mean(flow_network_sens_approach.boundary_val) #flow_network_sens_approach.boundary_val[0]

        # Nun das gleiche inverse Problem, wenn edge ids mit Messungen random gewählt werden
        from numpy.random import default_rng
        rng = default_rng()
        numbers = rng.choice(flow_network_ground_truth.nr_of_es, size=nr_of_meas, replace=False)  # Wähle die edge ids mit Messungen
        eid_meas_rand_approach = numbers

        inverse_model_rand_approach.edge_constraint_eid = np.array([eid_meas_rand_approach]).reshape(-1)
        inverse_model_rand_approach.nr_of_edge_constraints = np.size(inverse_model_rand_approach.edge_constraint_eid)
        inverse_model_rand_approach.edge_constraint_type = np.ones(inverse_model_rand_approach.nr_of_edge_constraints)
        inverse_model_rand_approach.edge_constraint_value = np.array(
            [flowrate_ground_truth[eid_meas_rand_approach]]).reshape(-1)
        inverse_model_rand_approach.edge_constraint_range_pm = np.zeros(inverse_model_rand_approach.nr_of_edge_constraints)
        inverse_model_rand_approach.edge_constraint_sigma = np.ones(
            inverse_model_rand_approach.nr_of_edge_constraints) * np.abs(flowrate_ground_truth[eid_max_sensitivity])

        inverse_model_rand_approach.update_cost()

        print("Solve the inverse problem and update the boundaries: ...")
        j = 0
        while inverse_model_rand_approach.f_h > (1e-5 * nr_of_meas) and j < 30001:
            inverse_model_rand_approach.update_state()
            flow_network_rand_approach.update_transmissibility()
            flow_network_rand_approach.update_blood_flow()
            inverse_model_rand_approach.update_cost()
            if j % 5 == 0:
                print(str(j) + " iterations done (f_H =", "%.2e" % inverse_model_rand_approach.f_h + ")")
            j += 1
            if True in (flow_network_rand_approach.diameter < 0):
                import sys

                sys.exit("Negative diameter detected")

        print(str(j - 1) + " iterations done (f_H =", "%.2e" % inverse_model_rand_approach.f_h + ")")
        print("Solve the inverse problem and update the boundaries: DONE")

        flow_rates_opt_random[:, realisation] = flow_network_rand_approach.flow_rate
        # Normalisiere boundary pressure mit mean pressure
        boundary_pressure_opt_random[:, realisation] = flow_network_rand_approach.boundary_val - \
                                                       np.mean(flow_network_rand_approach.boundary_val)

    # Differences between tuned and ground truth flow rates
    diff_flowrate_to_truth_sens = flow_rates_opt_sensitivity - flowrate_ground_truth.reshape(-1, 1)
    diff_flowrate_to_truth_rand = flow_rates_opt_random - flowrate_ground_truth.reshape(-1, 1)

    # Differences between tuned and ground truth boundary pressures
    diff_bound_to_truth_sens = boundary_pressure_opt_sensitivity - boundary_pressure_ground_truth.reshape(-1, 1)
    diff_bound_to_truth_rand = boundary_pressure_opt_random - boundary_pressure_ground_truth.reshape(-1, 1)

    # L2 norms of all flow rate differences
    norm_flowrate_to_truth_sens[nr_of_meas] = np.linalg.norm(diff_flowrate_to_truth_sens,axis=0).tolist()
    norm_flowrate_to_truth_rand[nr_of_meas] = np.linalg.norm(diff_flowrate_to_truth_rand,axis=0).tolist()

    norm_boundary_to_truth_sens[nr_of_meas] = np.linalg.norm(diff_bound_to_truth_sens,axis=0).tolist()
    norm_boundary_to_truth_rand[nr_of_meas] = np.linalg.norm(diff_bound_to_truth_rand,axis=0).tolist()

import json

with open('output/norm_flowrate_to_truth_sens.json', 'w') as fp:
    json.dump(norm_flowrate_to_truth_sens, fp)

with open('output/norm_flowrate_to_truth_rand.json', 'w') as fp:
    json.dump(norm_flowrate_to_truth_rand, fp)

with open('output/norm_boundary_to_truth_sens.json', 'w') as fp:
    json.dump(norm_boundary_to_truth_sens, fp)

with open('output/norm_boundary_to_truth_rand.json', 'w') as fp:
    json.dump(norm_boundary_to_truth_rand, fp)





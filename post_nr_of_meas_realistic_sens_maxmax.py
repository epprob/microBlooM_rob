import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



with open('output/realistic_box_200_onlySens_initialisation_0/norm_flowrate_to_truth_sens.json', 'r') as f:
    norm_flowrate_to_truth_sens = json.load(f)

with open('output/realistic_box_200_sensitivity_edgeremove/norm_flowrate_to_truth_sens_remove.json', 'r') as f:
    norm_flowrate_to_truth_remove = json.load(f)

with open('output/realistic_box_200_onlySens_initialisation_0/norm_boundary_to_truth_sens.json', 'r') as f:
    norm_boundary_to_truth_sens = json.load(f)

with open('output/realistic_box_200_sensitivity_edgeremove/norm_boundary_to_truth_sens_remove.json', 'r') as f:
    norm_boundary_to_truth_remove = json.load(f)

nr_of_samples = np.size(norm_flowrate_to_truth_sens['11'])

flowrate_norm_sens = np.hstack([norm_flowrate_to_truth_sens[i] for i in norm_flowrate_to_truth_sens])
boundary_norm_sens = np.hstack([norm_boundary_to_truth_sens[i] for i in norm_boundary_to_truth_sens])
nr_of_meas_sens = np.hstack([[i]*np.size(norm_flowrate_to_truth_sens[i]) for i in norm_flowrate_to_truth_sens])
strategy_sens = np.array(['sensitivity']*np.size(nr_of_meas_sens))

flowrate_norm_rand = np.hstack([norm_flowrate_to_truth_remove[i] for i in norm_flowrate_to_truth_remove])
boundary_norm_rand = np.hstack([norm_boundary_to_truth_remove[i] for i in norm_boundary_to_truth_remove])
nr_of_meas_rand = np.hstack([[i] * np.size(norm_flowrate_to_truth_remove[i]) for i in norm_flowrate_to_truth_remove])
strategy_rand = np.array(['remove']*np.size(nr_of_meas_rand))

fig, ax = plt.subplots(1, 1, figsize=(10,5))

df_plot = pd.DataFrame()
df_plot['flow rates error to ground-truth'] = np.append(flowrate_norm_sens,flowrate_norm_rand)
df_plot['boundary pressure error to ground-truth'] = np.append(boundary_norm_sens,boundary_norm_rand)
df_plot['Nr of measurements'] = np.append(nr_of_meas_sens, nr_of_meas_rand)
df_plot['strategy'] = np.append(strategy_sens, strategy_rand)


# sns.boxplot(df_plot, x = 'Nr of measurements', y='flow rates error to ground-truth', hue='strategy', ax=ax)

sns.boxplot(df_plot, x = 'Nr of measurements', y='flow rates error to ground-truth', hue='strategy', ax=ax, medianprops={'color': 'red', 'label': '_median_'})
median_colors = ['blue', 'orange']
median_lines = [line for line in ax.get_lines() if line.get_label() == '_median_']
for i, line in enumerate(median_lines):
    line.set_color(median_colors[i % len(median_colors)])

# sns.boxplot(df_plot, x = 'Nr of measurements', y='boundary pressure error to ground-truth', hue='strategy', ax=ax[1])

fig.suptitle('Testcase with realistic network 200*200*200, nr of samples = '+str(nr_of_samples))
ax.set_title('L2 norm flow rates to ground truth')
# ax[1].set_title('L2 norm boundary pressures to ground truth')

fig.savefig("output/realistic_box_200_sensitivity_edgeremove/comparison_sensitivity_edge_remove_sens.png", dpi=200)
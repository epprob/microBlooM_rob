import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



with open('output/norm_flowrate_to_truth_sens.json', 'r') as f:
    norm_flowrate_to_truth_sens = json.load(f)

with open('output/norm_flowrate_to_truth_maxmax.json', 'r') as f:
    norm_flowrate_to_truth_rand = json.load(f)

with open('output/norm_boundary_to_truth_sens.json', 'r') as f:
    norm_boundary_to_truth_sens = json.load(f)

with open('output/norm_boundary_to_truth_maxmax.json', 'r') as f:
    norm_boundary_to_truth_rand = json.load(f)

nr_of_samples = np.size(norm_flowrate_to_truth_sens['1'])

flowrate_norm_sens = np.hstack([norm_flowrate_to_truth_sens[i] for i in norm_flowrate_to_truth_sens])
boundary_norm_sens = np.hstack([norm_boundary_to_truth_sens[i] for i in norm_boundary_to_truth_sens])
nr_of_meas_sens = np.hstack([[i]*np.size(norm_flowrate_to_truth_sens[i]) for i in norm_flowrate_to_truth_sens])
strategy_sens = np.array(['sensitivity']*np.size(nr_of_meas_sens))

flowrate_norm_rand = np.hstack([norm_flowrate_to_truth_rand[i] for i in norm_flowrate_to_truth_rand])
boundary_norm_rand = np.hstack([norm_boundary_to_truth_rand[i] for i in norm_boundary_to_truth_rand])
nr_of_meas_rand = np.hstack([[i]*np.size(norm_flowrate_to_truth_rand[i]) for i in norm_flowrate_to_truth_rand])
strategy_rand = np.array(['max max']*np.size(nr_of_meas_rand))

fig, ax = plt.subplots(1, 2, figsize=(10,5))

df_plot = pd.DataFrame()
df_plot['flow rates error to ground-truth'] = np.append(flowrate_norm_sens,flowrate_norm_rand)
df_plot['boundary pressure error to ground-truth'] = np.append(boundary_norm_sens,boundary_norm_rand)
df_plot['Nr of measurements'] = np.append(nr_of_meas_sens, nr_of_meas_rand)
df_plot['strategy'] = np.append(strategy_sens, strategy_rand)


sns.boxplot(df_plot, x = 'Nr of measurements', y='flow rates error to ground-truth', hue='strategy', ax=ax[0])

sns.boxplot(df_plot, x = 'Nr of measurements', y='boundary pressure error to ground-truth', hue='strategy', ax=ax[1])

fig.suptitle('Testcase with hexagonal network, nr of samples = '+str(nr_of_samples))
ax[0].set_title('L2 norm flow rates to ground truth')
ax[1].set_title('L2 norm boundary pressures to ground truth')

fig.savefig("output/comparison_to_groundtruth_distortion_std_2.png", dpi=200)
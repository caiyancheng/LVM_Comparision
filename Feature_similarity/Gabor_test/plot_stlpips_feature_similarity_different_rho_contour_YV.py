import sys
sys.path.append('E:\Py_codes\LVM_Comparision')
import matplotlib.pyplot as plt
import json
import numpy as np
import itertools
import os

ppd = 60

save_root_path = 'new_contour_plots/stlpips/different_rho_YV'
os.makedirs(save_root_path, exist_ok=True)

json_data_path = rf'new_data_logs/stlpips/different_rho_YV/stlpips_test_on_gabors_different_rho_contour_plot_ppd_{ppd}_YV_final.json'
with open(json_data_path, 'r') as fp:
    plot_json_data = json.load(fp)

plot_rho_matrix = np.array(plot_json_data['rho_matrix'])[0]
plot_contrast_matrix = np.array(plot_json_data['contrast_matrix'])[0]
plot_loss_fn_alex_matrix = np.array(plot_json_data['loss_fn_alex_matrix'])[0]
plot_loss_fn_vgg_matrix = np.array(plot_json_data['loss_fn_vgg_matrix'])[0]
x_rho_ticks = [0.5, 1, 2, 4, 8, 16, 32]
y_contrast_ticks = [0.001, 0.01, 0.1, 0.2]
y_sensitivity_ticks = [5, 10, 100, 1000]

fig, axs = plt.subplots(2, 2, figsize=(15, 7))
plot_fn_list = [plot_loss_fn_alex_matrix,
                plot_loss_fn_vgg_matrix]
title_name_list = ['STLPIPS - AlexNet - YV', 'STLPIPS - VggNet - YV']

axs_all = itertools.chain(*axs)
index = 0
for ax in axs_all:
    if index < len(title_name_list):
        cs = ax.contour(plot_rho_matrix, plot_contrast_matrix, plot_fn_list[index], levels=20)
        ax.set_title(title_name_list[index])
        ax.set_yticks(y_contrast_ticks)
        ax.set_ylabel('Contrast')
    else:
        cs = ax.contour(plot_rho_matrix, 1 / plot_contrast_matrix, plot_fn_list[index - len(title_name_list)], levels=20)
        ax.set_title(title_name_list[index - len(title_name_list)])
        ax.set_yticks(y_sensitivity_ticks)
        ax.set_ylabel('Sensitivity')

    cbar = fig.colorbar(cs, ax=ax, orientation='vertical')
    # cbar.set_label('Value')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(x_rho_ticks)
    ax.set_xticklabels(x_rho_ticks)
    ax.set_xlabel('Spatial Frequency (cpd)')
    index += 1

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(save_root_path, f'STLPIPS_ppd_{ppd}_contour_plot_YV'))
X = 1
import matplotlib.pyplot as plt
import json
import numpy as np
import itertools
import os
ppd = 60

save_root_path = 'new_contour_plots/lpips/different_luminance'
os.makedirs(save_root_path, exist_ok=True)

json_data_path = rf'new_data_logs/lpips/different_luminance/lpips_test_on_gabors_different_L_b_contour_plot_ppd_{ppd}_final.json'
with open(json_data_path, 'r') as fp:
    plot_json_data = json.load(fp)

plot_L_b_matrix = np.array(plot_json_data['L_b_matrix'])[0]
plot_contrast_matrix = np.array(plot_json_data['contrast_matrix'])[0]
plot_loss_fn_alex_matrix = np.array(plot_json_data['loss_fn_alex_matrix'])[0]
plot_loss_fn_vgg_matrix = np.array(plot_json_data['loss_fn_vgg_matrix'])[0]
plot_loss_fn_squeeze_matrix = np.array(plot_json_data['loss_fn_squeeze_matrix'])[0]
x_luminance_ticks = [0.1, 1, 10, 100, 200]
y_contrast_ticks = [0.001, 0.01, 0.1, 1]
y_sensitivity_ticks = [1, 10, 100, 1000]

fig, axs = plt.subplots(2, 3, figsize=(15, 7))
plot_fn_list = [plot_loss_fn_alex_matrix,
                plot_loss_fn_vgg_matrix,
                plot_loss_fn_squeeze_matrix]
title_name_list = ['LPIPS - AlexNet', 'LPIPS - VggNet', 'LPIPS - SqueezeNet']

axs_all = itertools.chain(*axs)
index = 0
for ax in axs_all:
    if index < 3:
        cs = ax.contour(plot_L_b_matrix, plot_contrast_matrix, plot_fn_list[index], levels=20)
        ax.set_title(title_name_list[index])
        ax.set_yticks(y_contrast_ticks)
        ax.set_ylabel('Contrast')
    else:
        cs = ax.contour(plot_L_b_matrix, 1 / plot_contrast_matrix, plot_fn_list[index - 3], levels=20)
        ax.set_title(title_name_list[index - 3])
        ax.set_yticks(y_sensitivity_ticks)
        ax.set_ylabel('Sensitivity')

    cbar = fig.colorbar(cs, ax=ax, orientation='vertical')
    # cbar.set_label('Value')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(x_luminance_ticks)
    ax.set_xticklabels(x_luminance_ticks)
    ax.set_xlabel('Luminance (nits) Gamma-encode')
    index += 1

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(save_root_path, f'LPIPS_ppd_{ppd}_contour_plot'))
X = 1

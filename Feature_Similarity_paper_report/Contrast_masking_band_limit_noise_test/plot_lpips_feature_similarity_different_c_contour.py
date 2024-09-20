import matplotlib.pyplot as plt
import json
import numpy as np
import itertools
import os

ppd = 60

save_root_path = 'new_contour_plots/lpips/different_c'
os.makedirs(save_root_path, exist_ok=True)

json_data_path = rf'new_data_logs/lpips/different_c/lpips_test_on_contrast_masking_different_c_contour_plot_ppd_{ppd}_final.json'
with open(json_data_path, 'r') as fp:
    plot_json_data = json.load(fp)

plot_contrast_mask_matrix = np.array(plot_json_data['contrast_mask_matrix'])[0]
plot_contrast_test_matrix = np.array(plot_json_data['contrast_test_matrix'])[0]
plot_loss_fn_alex_matrix = np.array(plot_json_data['loss_fn_alex_matrix'])[0]
plot_loss_fn_vgg_matrix = np.array(plot_json_data['loss_fn_vgg_matrix'])[0]
plot_loss_fn_squeeze_matrix = np.array(plot_json_data['loss_fn_squeeze_matrix'])[0]
x_contrast_mask_ticks = [0.01, 0.1]
y_contrast_test_ticks = [0.01, 0.1]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
plot_fn_list = [plot_loss_fn_alex_matrix,
                plot_loss_fn_vgg_matrix,
                plot_loss_fn_squeeze_matrix]
title_name_list = ['LPIPS - AlexNet', 'LPIPS - VggNet', 'LPIPS - SqueezeNet']

axs_all = itertools.chain(axs)
index = 0
for ax in axs_all:
    cs = ax.contour(plot_contrast_mask_matrix, plot_contrast_test_matrix, plot_fn_list[index], levels=20)
    ax.set_title(title_name_list[index])
    ax.set_yticks(y_contrast_test_ticks)
    ax.set_ylabel('Test Contrast')

    cbar = fig.colorbar(cs, ax=ax, orientation='vertical')
    # cbar.set_label('Value')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(x_contrast_mask_ticks)
    ax.set_xlabel('Mask Contrast')
    index += 1

plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(save_root_path, f'LPIPS_ppd_{ppd}_contour_plot'))
X = 1

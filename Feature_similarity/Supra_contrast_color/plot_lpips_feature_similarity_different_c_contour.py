import matplotlib.pyplot as plt
import json
import numpy as np
import itertools
import os

ppd = 60

save_root_path = 'new_contour_plots/lpips/different_c'
os.makedirs(save_root_path, exist_ok=True)

json_data_path = rf'new_data_logs/lpips/different_c/lpips_test_on_supra_contrast_color_different_c_contour_plot_ppd_{ppd}_final.json'
with open(json_data_path, 'r') as fp:
    plot_json_data = json.load(fp)

plot_contrast_test_matrix = np.array(plot_json_data['contrast_test_matrix'])[0]
plot_color_direction_matrix = np.array(plot_json_data['color_direction_matrix'])[0]
color_direction_map = ['ach', 'rg', 'yv']
plot_loss_fn_alex_matrix = np.array(plot_json_data['loss_fn_alex_matrix'])[0]
plot_loss_fn_vgg_matrix = np.array(plot_json_data['loss_fn_vgg_matrix'])[0]
plot_loss_fn_squeeze_matrix = np.array(plot_json_data['loss_fn_squeeze_matrix'])[0]
x_color_direction_ticks = [0, 1, 2]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
contrast_test_list = plot_contrast_test_matrix[0]
title_name_list = ['LPIPS - AlexNet', 'LPIPS - VggNet', 'LPIPS - SqueezeNet']

for contrast_test_index in range(len(contrast_test_list)):
    contrast_test_value = contrast_test_list[contrast_test_index]
    X_color_direction = plot_color_direction_matrix[:, contrast_test_index]
    Y_loss_fn_alex_score = plot_loss_fn_alex_matrix[:, contrast_test_index]
    Y_loss_fn_vgg_score = plot_loss_fn_vgg_matrix[:, contrast_test_index]
    Y_loss_fn_squeeze_score = plot_loss_fn_squeeze_matrix[:, contrast_test_index]

    plot_Y_score_list = [Y_loss_fn_alex_score, Y_loss_fn_vgg_score, Y_loss_fn_squeeze_score]
    index = 0
    axs_all = itertools.chain(axs)
    for ax in axs_all:
        ax.plot(X_color_direction, plot_Y_score_list[index], label=f'c = {contrast_test_value}')
        index += 1
index = 0
axs_all = itertools.chain(axs)
for ax in axs_all:
    ax.set_title(title_name_list[index])
    ax.set_xlabel('Color Direction')
    ax.set_ylabel('Similarity Score')
    ax.set_xticks(x_color_direction_ticks)
    ax.set_xticklabels(color_direction_map)
    index += 1
plt.tight_layout()
plt.savefig(os.path.join(save_root_path, f'LPIPS_ppd_{ppd}_contour_plot'))

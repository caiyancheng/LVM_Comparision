import sys
sys.path.append('E:\Py_codes\LVM_Comparision')
import matplotlib.pyplot as plt
import json
import numpy as np
import itertools
import os
from tqdm import tqdm
ppd = 60

save_root_path = 'new_contour_plots/dinov2/different_rho'
os.makedirs(save_root_path, exist_ok=True)

json_data_path = rf'new_data_logs/dinov2/different_rho/dinov2_test_on_gabors_different_rho_contour_plot_ppd_{ppd}_final.json'
with open(json_data_path, 'r') as fp:
    plot_json_data = json.load(fp)

backbone_name_list = plot_json_data['backbone_name']
plot_rho_matrix_list = plot_json_data['rho_matrix']
plot_contrast_matrix_list = plot_json_data['contrast_matrix']
plot_final_feature_L1_similarity_matrix_list = plot_json_data['final_feature_L1_similarity_matrix']
plot_final_feature_L2_similarity_matrix_list = plot_json_data['final_feature_L2_similarity_matrix']
plot_final_feature_cos_similarity_matrix_list = plot_json_data['final_feature_cos_similarity_matrix']
plot_intermediate_feature_L1_similarity_matrix_list = plot_json_data['intermediate_feature_L1_similarity_matrix']
plot_intermediate_feature_L2_similarity_matrix_list = plot_json_data['intermediate_feature_L2_similarity_matrix']
plot_intermediate_feature_cos_similarity_matrix_list = plot_json_data['intermediate_feature_cos_similarity_matrix']
x_rho_ticks = [0.5, 1, 2, 4, 8, 16, 32]
y_contrast_ticks = [0.001, 0.01, 0.1, 1]
y_sensitivity_ticks = [1, 10, 100, 1000]

for backbone_index in tqdm(range(len(backbone_name_list))):
    fig, axs = plt.subplots(4, 3, figsize=(15, 10))
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    backbone_name = backbone_name_list[backbone_index]
    plot_rho_matrix = np.array(plot_rho_matrix_list[backbone_index])
    plot_contrast_matrix = np.array(plot_contrast_matrix_list[backbone_index])
    plot_final_feature_L1_similarity_matrix = np.array(plot_final_feature_L1_similarity_matrix_list[backbone_index])
    plot_final_feature_L2_similarity_matrix = np.array(plot_final_feature_L2_similarity_matrix_list[backbone_index])
    plot_final_feature_cos_similarity_matrix = np.array(plot_final_feature_cos_similarity_matrix_list[backbone_index])
    plot_intermediate_feature_L1_similarity_matrix = np.array(
        plot_intermediate_feature_L1_similarity_matrix_list[backbone_index])
    plot_intermediate_feature_L2_similarity_matrix = np.array(
        plot_intermediate_feature_L2_similarity_matrix_list[backbone_index])
    plot_intermediate_feature_cos_similarity_matrix = np.array(
        plot_intermediate_feature_cos_similarity_matrix_list[backbone_index])

    plot_similarity_list = [plot_final_feature_L1_similarity_matrix,
                            plot_final_feature_L2_similarity_matrix,
                            plot_final_feature_cos_similarity_matrix,
                            plot_intermediate_feature_L1_similarity_matrix,
                            plot_intermediate_feature_L2_similarity_matrix,
                            plot_intermediate_feature_cos_similarity_matrix]
    title_name_list = [f'{backbone_name} - L1 similarity - final feature',
                       f'{backbone_name} - L2 similarity - final feature',
                       f'{backbone_name} - cos similarity - final feature',
                       f'{backbone_name} - L1 similarity - intermediate feature',
                       f'{backbone_name} - L2 similarity - intermediate feature',
                       f'{backbone_name} - cos similarity - intermediate feature']

    axs_all = itertools.chain(*axs)
    index = 0
    for ax in axs_all:
        if index < 6:
            cs = ax.contour(plot_rho_matrix, plot_contrast_matrix, plot_similarity_list[index], levels=20)
            ax.set_title(title_name_list[index])
            ax.set_yticks(y_contrast_ticks)
            ax.set_ylabel('Contrast')
        else:
            cs = ax.contour(plot_rho_matrix, 1 / plot_contrast_matrix, plot_similarity_list[index - 6], levels=20)
            ax.set_title(title_name_list[index - 6])
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
    plt.savefig(os.path.join(save_root_path, f'dinov2_ppd_{ppd}_{backbone_name}_contour_plot'))
    X = 1


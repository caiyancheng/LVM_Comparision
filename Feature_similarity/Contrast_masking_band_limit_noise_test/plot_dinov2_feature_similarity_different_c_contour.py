import matplotlib.pyplot as plt
import json
import numpy as np
import itertools
import os
from tqdm import tqdm

ppd = 60

save_root_path = 'new_contour_plots/dinov2/different_c'
os.makedirs(save_root_path, exist_ok=True)

json_data_path = rf'new_data_logs/dinov2/different_c/dinov2_test_on_contrast_masking_different_c_contour_plot_ppd_{ppd}_final.json'
with open(json_data_path, 'r') as fp:
    plot_json_data = json.load(fp)

backbone_name_list = plot_json_data['backbone_name']
plot_contrast_mask_matrix_list = plot_json_data['contrast_mask_matrix']
plot_contrast_test_matrix_list = plot_json_data['contrast_test_matrix']
plot_final_feature_L1_similarity_matrix_list = plot_json_data['final_feature_L1_similarity_matrix']
plot_final_feature_L2_similarity_matrix_list = plot_json_data['final_feature_L2_similarity_matrix']
plot_final_feature_cos_similarity_matrix_list = plot_json_data['final_feature_cos_similarity_matrix']
plot_intermediate_feature_L1_similarity_matrix_list = plot_json_data['intermediate_feature_L1_similarity_matrix']
plot_intermediate_feature_L2_similarity_matrix_list = plot_json_data['intermediate_feature_L2_similarity_matrix']
plot_intermediate_feature_cos_similarity_matrix_list = plot_json_data['intermediate_feature_cos_similarity_matrix']
x_contrast_mask_ticks = [0.01, 0.1]
y_contrast_test_ticks = [0.01, 0.1]

for backbone_index in tqdm(range(len(backbone_name_list))):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    backbone_name = backbone_name_list[backbone_index]
    plot_contrast_mask_matrix = np.array(plot_contrast_mask_matrix_list[backbone_index])
    plot_contrast_test_matrix = np.array(plot_contrast_test_matrix_list[backbone_index])
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
        cs = ax.contour(plot_contrast_mask_matrix, plot_contrast_test_matrix, plot_similarity_list[index], levels=20)
        ax.set_title(title_name_list[index])
        ax.set_yticks(y_contrast_test_ticks)
        ax.set_ylabel('Test Contrast')

        cbar = fig.colorbar(cs, ax=ax, orientation='vertical')
        # cbar.set_label('Value')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(x_contrast_mask_ticks)
        ax.set_xticklabels(x_contrast_mask_ticks)
        ax.set_xlabel('Mask Contrast')
        index += 1

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(save_root_path, f'dinov2_ppd_{ppd}_{backbone_name}_contour_plot'))


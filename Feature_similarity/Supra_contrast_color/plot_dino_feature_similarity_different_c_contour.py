import matplotlib.pyplot as plt
import json
import numpy as np
import itertools
import os
from tqdm import tqdm

ppd = 60

save_root_path = 'new_contour_plots/dino/different_c'
os.makedirs(save_root_path, exist_ok=True)

json_data_path = rf'new_data_logs/dino/different_c/dino_test_on_supra_contrast_color_different_c_contour_plot_ppd_{ppd}_final.json'
with open(json_data_path, 'r') as fp:
    plot_json_data = json.load(fp)

backbone_name_list = plot_json_data['backbone_name']
plot_contrast_test_matrix_list = plot_json_data['contrast_test_matrix']
plot_color_direction_matrix_list = plot_json_data['color_direction_matrix']
color_direction_map = ['ach', 'rg', 'yv']
plot_final_feature_L1_similarity_matrix_list = plot_json_data['final_feature_L1_similarity_matrix']
plot_final_feature_L2_similarity_matrix_list = plot_json_data['final_feature_L2_similarity_matrix']
plot_final_feature_cos_similarity_matrix_list = plot_json_data['final_feature_cos_similarity_matrix']
plot_intermediate_feature_L1_similarity_matrix_list = plot_json_data['intermediate_feature_L1_similarity_matrix']
plot_intermediate_feature_L2_similarity_matrix_list = plot_json_data['intermediate_feature_L2_similarity_matrix']
plot_intermediate_feature_cos_similarity_matrix_list = plot_json_data['intermediate_feature_cos_similarity_matrix']
x_color_direction_ticks = [0,1,2]
# y_contrast_ticks = [0.001, 0.01, 0.1, 1]
# y_sensitivity_ticks = [1, 10, 100, 1000]

for backbone_index in tqdm(range(len(backbone_name_list))):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    backbone_name = backbone_name_list[backbone_index]
    plot_contrast_test_matrix = np.array(plot_contrast_test_matrix_list[backbone_index])
    plot_color_direction_matrix = np.array(plot_color_direction_matrix_list[backbone_index])
    plot_final_feature_L1_similarity_matrix = np.array(plot_final_feature_L1_similarity_matrix_list[backbone_index])
    plot_final_feature_L2_similarity_matrix = np.array(plot_final_feature_L2_similarity_matrix_list[backbone_index])
    plot_final_feature_cos_similarity_matrix = np.array(plot_final_feature_cos_similarity_matrix_list[backbone_index])
    plot_intermediate_feature_L1_similarity_matrix = np.array(
        plot_intermediate_feature_L1_similarity_matrix_list[backbone_index])
    plot_intermediate_feature_L2_similarity_matrix = np.array(
        plot_intermediate_feature_L2_similarity_matrix_list[backbone_index])
    plot_intermediate_feature_cos_similarity_matrix = np.array(
        plot_intermediate_feature_cos_similarity_matrix_list[backbone_index])

    contrast_test_list = plot_contrast_test_matrix[0]
    title_name_list = [f'{backbone_name} - L1 similarity - final feature',
                       f'{backbone_name} - L2 similarity - final feature',
                       f'{backbone_name} - cos similarity - final feature',
                       f'{backbone_name} - L1 similarity - intermediate feature',
                       f'{backbone_name} - L2 similarity - intermediate feature',
                       f'{backbone_name} - cos similarity - intermediate feature']
    for contrast_test_index in range(len(contrast_test_list)):
        contrast_test_value = contrast_test_list[contrast_test_index]
        # print('Contrast Value:', contrast_test_value)
        X_color_direction = plot_color_direction_matrix[:,contrast_test_index]
        Y_final_feature_L1_similarity = plot_final_feature_L1_similarity_matrix[:,contrast_test_index]
        Y_final_feature_L2_similarity = plot_final_feature_L2_similarity_matrix[:, contrast_test_index]
        Y_final_feature_cos_similarity = plot_final_feature_cos_similarity_matrix[:, contrast_test_index]
        Y_intermediate_feature_L1_similarity = plot_intermediate_feature_L1_similarity_matrix[:, contrast_test_index]
        Y_intermediate_feature_L2_similarity = plot_intermediate_feature_L2_similarity_matrix[:, contrast_test_index]
        Y_intermediate_feature_cos_similarity = plot_intermediate_feature_cos_similarity_matrix[:, contrast_test_index]

        plot_Y_similarity_list = [Y_final_feature_L1_similarity,
                                  Y_final_feature_L2_similarity,
                                  Y_final_feature_cos_similarity,
                                  Y_intermediate_feature_L1_similarity,
                                  Y_intermediate_feature_L2_similarity,
                                  Y_intermediate_feature_cos_similarity]
        index = 0
        axs_all = itertools.chain(*axs)
        for ax in axs_all:
            ax.plot(X_color_direction, plot_Y_similarity_list[index], label=f'c = {contrast_test_value}')
            # ax.legend()
            index += 1
    index = 0
    axs_all = itertools.chain(*axs)
    for ax in axs_all:
        ax.set_title(title_name_list[index])
        ax.set_xlabel('Color Direction')
        ax.set_ylabel('Similarity Score')
        ax.set_xticks(x_color_direction_ticks)
        ax.set_xticklabels(color_direction_map)
        index += 1

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(save_root_path, f'dino_ppd_{ppd}_{backbone_name}_contour_plot'))


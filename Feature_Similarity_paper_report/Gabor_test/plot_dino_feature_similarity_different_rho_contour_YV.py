import sys
sys.path.append('E:\Py_codes\LVM_Comparision')
import matplotlib.pyplot as plt
import json
import numpy as np
import math
import os
from tqdm import tqdm
ppd = 60

save_root_path = 'new_contour_plots/dino/different_rho_YV'
os.makedirs(save_root_path, exist_ok=True)

json_data_path = rf'new_data_logs/dino/different_rho_YV/dino_test_on_gabors_different_rho_contour_plot_ppd_{ppd}_YV_final.json'
with open(json_data_path, 'r') as fp:
    plot_json_data = json.load(fp)
castleCSF_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/castleCSF_rho_sensitivity_data_YV.json'
with open(castleCSF_result_json, 'r') as fp:
    castleCSF_result_data = json.load(fp)
castleCSF_result_rho_list = castleCSF_result_data['rho_list']
castleCSF_result_sensitivity_list = castleCSF_result_data['sensitivity_list']

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
    backbone_name = backbone_name_list[backbone_index]
    real_save_path = os.path.join(save_root_path, backbone_name)
    os.makedirs(real_save_path, exist_ok=True)

    plot_rho_matrix = np.array(plot_rho_matrix_list[backbone_index])
    plot_contrast_matrix = np.array(plot_contrast_matrix_list[backbone_index])
    plot_final_feature_L1_similarity_matrix = np.array(plot_final_feature_L1_similarity_matrix_list[backbone_index])
    plot_final_feature_L2_similarity_matrix = np.array(plot_final_feature_L2_similarity_matrix_list[backbone_index])
    plot_final_feature_cos_similarity_matrix = np.array(plot_final_feature_cos_similarity_matrix_list[backbone_index])
    if backbone_name.startswith('dino_vit'):
        plot_intermediate_feature_L1_similarity_matrix = np.array(
            plot_intermediate_feature_L1_similarity_matrix_list[backbone_index])
        plot_intermediate_feature_L2_similarity_matrix = np.array(
            plot_intermediate_feature_L2_similarity_matrix_list[backbone_index])
        plot_intermediate_feature_cos_similarity_matrix = np.array(
            plot_intermediate_feature_cos_similarity_matrix_list[backbone_index])
        plot_figure_data_matrix_list = [plot_final_feature_L1_similarity_matrix,
                                        plot_final_feature_L2_similarity_matrix,
                                        plot_final_feature_cos_similarity_matrix,
                                        plot_intermediate_feature_L1_similarity_matrix,
                                        plot_intermediate_feature_L2_similarity_matrix,
                                        plot_intermediate_feature_cos_similarity_matrix]
        plot_figure_name_list = [f'{backbone_name} - L1 similarity - final feature',
                                 f'{backbone_name} - L2 similarity - final feature',
                                 f'{backbone_name} - cos similarity - final feature',
                                 f'{backbone_name} - L1 similarity - intermediate feature',
                                 f'{backbone_name} - L2 similarity - intermediate feature',
                                 f'{backbone_name} - cos similarity - intermediate feature']
    else:
        plot_figure_data_matrix_list = [plot_final_feature_L1_similarity_matrix,
                                        plot_final_feature_L2_similarity_matrix,
                                        plot_final_feature_cos_similarity_matrix]
        plot_figure_name_list = [f'{backbone_name} - L1 similarity - final feature',
                                 f'{backbone_name} - L2 similarity - final feature',
                                 f'{backbone_name} - cos similarity - final feature']

    for figure_index in range(len(plot_figure_name_list)):
        plt.figure(figsize=(5, 5))
        plt.contour(plot_rho_matrix, 1 / plot_contrast_matrix, plot_figure_data_matrix_list[figure_index], levels=20)
        plt.plot(castleCSF_result_rho_list, castleCSF_result_sensitivity_list, 'r', linestyle='--', linewidth=2,
                 label='castleCSF prediction (RG)')
        plt.xlabel('Stimulus Spatial Frequency (cpd)', fontsize=12)
        plt.ylabel('Sensitivity', fontsize=12)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([0.5, 32])
        plt.xticks(x_rho_ticks, x_rho_ticks)
        plt.yticks(y_sensitivity_ticks, y_sensitivity_ticks)
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(real_save_path, plot_figure_name_list[figure_index]))
        plt.close()


import matplotlib.pyplot as plt
import json
import numpy as np
import itertools
import os
from tqdm import tqdm

ppd = 60

save_root_path = 'new_contour_plots/openclip/different_c'
os.makedirs(save_root_path, exist_ok=True)

json_data_path = rf'new_data_logs/openclip/different_c/openclip_test_on_contrast_masking_different_c_contour_plot_ppd_{ppd}_final.json'
with open(json_data_path, 'r') as fp:
    plot_json_data = json.load(fp)
foley_result_json = r'E:\Py_codes\LVM_Comparision\Matlab_CSF_plot/foley_contrast_masking_data_gabor.json'
with open(foley_result_json, 'r') as fp:
    foley_result_data = json.load(fp)
foley_result_x_mask_contrast_list = foley_result_data['mask_contrast_list']
foley_result_y_test_contrast_list = foley_result_data['test_contrast_list']

clip_model_name_list = plot_json_data['clip_model_name']
clip_model_trainset_list = plot_json_data['clip_model_trainset']
plot_contrast_mask_matrix_list = plot_json_data['contrast_mask_matrix']
plot_contrast_test_matrix_list = plot_json_data['contrast_test_matrix']
plot_final_feature_L1_similarity_matrix_list = plot_json_data['final_feature_L1_similarity_matrix']
plot_final_feature_L2_similarity_matrix_list = plot_json_data['final_feature_L2_similarity_matrix']
plot_final_feature_cos_similarity_matrix_list = plot_json_data['final_feature_cos_similarity_matrix']
x_contrast_mask_ticks = [0.01, 0.1]
y_contrast_test_ticks = [0.01, 0.1]

for clip_model_index in tqdm(range(len(clip_model_name_list))):
    clip_model_name = clip_model_name_list[clip_model_index]
    clip_model_trainset = clip_model_trainset_list[clip_model_index]
    real_save_path = os.path.join(save_root_path, clip_model_name + ' - ' + clip_model_trainset)
    os.makedirs(real_save_path, exist_ok=True)

    plot_contrast_mask_matrix = np.array(plot_contrast_mask_matrix_list[clip_model_index])
    plot_contrast_test_matrix = np.array(plot_contrast_test_matrix_list[clip_model_index])
    plot_final_feature_L1_similarity_matrix = np.array(plot_final_feature_L1_similarity_matrix_list[clip_model_index])
    plot_final_feature_L2_similarity_matrix = np.array(plot_final_feature_L2_similarity_matrix_list[clip_model_index])
    plot_final_feature_cos_similarity_matrix = np.array(plot_final_feature_cos_similarity_matrix_list[clip_model_index])

    plot_figure_data_matrix_list = [plot_final_feature_L1_similarity_matrix,
                                    plot_final_feature_L2_similarity_matrix,
                                    plot_final_feature_cos_similarity_matrix]
    plot_figure_name_list = [f'{clip_model_name} - {clip_model_trainset} - L1 similarity - final feature',
                             f'{clip_model_name} - {clip_model_trainset} - L2 similarity - final feature',
                             f'{clip_model_name} - {clip_model_trainset} - cos similarity - final feature']

    for figure_index in range(len(plot_figure_name_list)):
        plt.figure(figsize=(5, 5))
        plt.contour(plot_contrast_mask_matrix, plot_contrast_test_matrix, plot_figure_data_matrix_list[figure_index],
                    levels=20)
        plt.plot(foley_result_x_mask_contrast_list, foley_result_y_test_contrast_list, 'r', linestyle='--', linewidth=2,
                 label='Human Results')
        plt.xlabel('Mask Contrast', fontsize=12)
        plt.ylabel('Test Contrast', fontsize=12)
        plt.xscale('log')
        plt.yscale('log')
        plt.xticks(x_contrast_mask_ticks, x_contrast_mask_ticks)
        plt.yticks(y_contrast_test_ticks, y_contrast_test_ticks)
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(real_save_path, plot_figure_name_list[figure_index]))
        plt.close()


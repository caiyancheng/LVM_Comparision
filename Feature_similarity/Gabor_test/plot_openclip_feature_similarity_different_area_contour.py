import sys
sys.path.append('E:\Py_codes\LVM_Comparision')
import matplotlib.pyplot as plt
import json
import numpy as np
import itertools
import os
from tqdm import tqdm

ppd = 60

save_root_path = 'new_contour_plots/openclip/different_area'
os.makedirs(save_root_path, exist_ok=True)

json_data_path = rf'new_data_logs/openclip/different_area/openclip_test_on_gabors_different_area_contour_plot_ppd_{ppd}_final.json'
with open(json_data_path, 'r') as fp:
    plot_json_data = json.load(fp)

clip_model_name_list = plot_json_data['clip_model_name']
clip_model_trainset_list = plot_json_data['clip_model_trainset']
plot_radius_matrix_list = plot_json_data['radius_matrix']
plot_area_matrix_list = plot_json_data['area_matrix']
plot_contrast_matrix_list = plot_json_data['contrast_matrix']
plot_final_feature_L1_similarity_matrix_list = plot_json_data['final_feature_L1_similarity_matrix']
plot_final_feature_L2_similarity_matrix_list = plot_json_data['final_feature_L2_similarity_matrix']
plot_final_feature_cos_similarity_matrix_list = plot_json_data['final_feature_cos_similarity_matrix']
x_area_ticks = [0.01, 0.1, 1, 10]
y_contrast_ticks = [0.001, 0.01, 0.1, 1]
y_sensitivity_ticks = [1, 10, 100, 1000]

for clip_model_index in tqdm(range(len(clip_model_name_list))):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    clip_model_name = clip_model_name_list[clip_model_index]
    clip_model_trainset = clip_model_trainset_list[clip_model_index]
    plot_area_matrix = np.array(plot_area_matrix_list[clip_model_index])
    plot_contrast_matrix = np.array(plot_contrast_matrix_list[clip_model_index])
    plot_final_feature_L1_similarity_matrix = np.array(plot_final_feature_L1_similarity_matrix_list[clip_model_index])
    plot_final_feature_L2_similarity_matrix = np.array(plot_final_feature_L2_similarity_matrix_list[clip_model_index])
    plot_final_feature_cos_similarity_matrix = np.array(plot_final_feature_cos_similarity_matrix_list[clip_model_index])

    plot_similarity_list = [plot_final_feature_L1_similarity_matrix,
                            plot_final_feature_L2_similarity_matrix,
                            plot_final_feature_cos_similarity_matrix]
    title_name_list = [f'{clip_model_name} - {clip_model_trainset} - L1 similarity - final feature',
                       f'{clip_model_name} - {clip_model_trainset} - L2 similarity - final feature',
                       f'{clip_model_name} - {clip_model_trainset} - cos similarity - final feature']

    axs_all = itertools.chain(*axs)
    index = 0
    for ax in axs_all:
        if index < 3:
            cs = ax.contour(plot_area_matrix, plot_contrast_matrix, plot_similarity_list[index], levels=20)
            ax.set_title(title_name_list[index])
            ax.set_yticks(y_contrast_ticks)
            ax.set_ylabel('Contrast')
        else:
            cs = ax.contour(plot_area_matrix, 1 / plot_contrast_matrix, plot_similarity_list[index - 6], levels=20)
            ax.set_title(title_name_list[index - 6])
            ax.set_yticks(y_sensitivity_ticks)
            ax.set_ylabel('Sensitivity')

        cbar = fig.colorbar(cs, ax=ax, orientation='vertical')
        # cbar.set_label('Value')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(x_area_ticks)
        ax.set_xticklabels(x_area_ticks)
        ax.set_xlabel('Stimulus Area (degree^2)')
        index += 1

    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(save_root_path, f'openclip_ppd_{ppd}_{clip_model_name}_{clip_model_trainset}_contour_plot'))
    X = 1


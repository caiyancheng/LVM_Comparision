import numpy as np
import torch
from Supra_contrast_color_generator.generate_plot_supra_contrast_color import generate_supra_contrast_color
import torch.nn.functional as F
import pandas as pd
import json
from tqdm import tqdm
import os
from display_encoding import display_encode
display_encode_tool = display_encode(400, 2.2)

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt

# Only test cpd right now
# Dinov2 input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/dinov2/different_c'
os.makedirs(save_root_path, exist_ok=True)
all_backbone_list = ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14',
                     'dinov2_vits14_reg', 'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg']
# all_backbone_list = ['dinov2_vits14']

default_W = 224
default_H = 224
default_rho = 1
default_L_b = 100
contrast_test_list = np.logspace(np.log10(0.01), np.log10(0.2), 20)
default_ppd = 60
color_direction_list = [0, 1, 2] #['ach', 'rg', 'yv']
color_direction_map = ['ach', 'rg', 'yv']

csv_data = {}
csv_data['backbone_name'] = []
csv_data['contrast_test'] = []
csv_data['color_direction'] = []
csv_data['final_feature_L1_similarity'] = []
csv_data['final_feature_L2_similarity'] = []
csv_data['final_feature_cos_similarity'] = []
csv_data['intermediate_feature_L1_similarity'] = []
csv_data['intermediate_feature_L2_similarity'] = []
csv_data['intermediate_feature_cos_similarity'] = []

json_plot_data = {}
json_plot_data['backbone_name'] = []
json_plot_data['contrast_test_matrix'] = []
json_plot_data['color_direction_matrix'] = []
json_plot_data['final_feature_L1_similarity_matrix'] = []
json_plot_data['final_feature_L2_similarity_matrix'] = []
json_plot_data['final_feature_cos_similarity_matrix'] = []
json_plot_data['intermediate_feature_L1_similarity_matrix'] = []
json_plot_data['intermediate_feature_L2_similarity_matrix'] = []
json_plot_data['intermediate_feature_cos_similarity_matrix'] = []

for backbone_name in tqdm(all_backbone_list):
    json_plot_data['backbone_name'].append(backbone_name)
    plot_contrast_test_matrix = np.zeros([len(color_direction_list), len(contrast_test_list)])
    plot_color_direction_matrix = np.zeros([len(color_direction_list), len(contrast_test_list)])
    plot_final_feature_L1_similarity_matrix = np.zeros([len(color_direction_list), len(contrast_test_list)])
    plot_final_feature_L2_similarity_matrix = np.zeros([len(color_direction_list), len(contrast_test_list)])
    plot_final_feature_cos_similarity_matrix = np.zeros([len(color_direction_list), len(contrast_test_list)])
    plot_intermediate_feature_L1_similarity_matrix = np.zeros([len(color_direction_list), len(contrast_test_list)])
    plot_intermediate_feature_L2_similarity_matrix = np.zeros([len(color_direction_list), len(contrast_test_list)])
    plot_intermediate_feature_cos_similarity_matrix = np.zeros([len(color_direction_list), len(contrast_test_list)])

    backbone_model = torch.hub.load('facebookresearch/dinov2', backbone_name)
    backbone_model.eval()
    backbone_model.cuda()

    for contrast_test_index in range(len(contrast_test_list)):
        contrast_test_value = contrast_test_list[contrast_test_index]
        for color_direction_index in range(len(color_direction_list)):
            color_direction_value = color_direction_list[color_direction_index]
            csv_data['backbone_name'].append(backbone_name)
            csv_data['contrast_test'].append(contrast_test_value)
            csv_data['color_direction'].append(color_direction_value)
            plot_contrast_test_matrix[color_direction_index, contrast_test_index] = contrast_test_value
            plot_color_direction_matrix[color_direction_index, contrast_test_index] = color_direction_value
            T_vid, R_vid = generate_supra_contrast_color(W=default_W, H=default_H, rho=default_rho, L_b=default_L_b,
                                                         contrast_test=contrast_test_value, ppd=default_ppd,
                                                         color_direction=color_direction_map[color_direction_value])
            T_vid_c = display_encode_tool.L2C(T_vid)
            R_vid_c = display_encode_tool.L2C(R_vid)
            T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
            R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...].cuda()
            test_feature = backbone_model(T_vid_ct)
            test_feature_intermediate = backbone_model.get_intermediate_layers(T_vid_ct, n=4)
            test_feature_intermediate = torch.stack(test_feature_intermediate)
            reference_feature = backbone_model(R_vid_ct)
            reference_feature_intermediate = backbone_model.get_intermediate_layers(R_vid_ct, n=4)
            reference_feature_intermediate = torch.stack(reference_feature_intermediate)

            L1_similarity = float(torch.norm(test_feature - reference_feature, p=1).cpu())
            L1_similarity_intermediate = float(
                torch.norm(test_feature_intermediate - reference_feature_intermediate, p=1).cpu())
            csv_data['final_feature_L1_similarity'].append(L1_similarity)
            csv_data['intermediate_feature_L1_similarity'].append(L1_similarity_intermediate)
            plot_final_feature_L1_similarity_matrix[color_direction_index, contrast_test_index] = L1_similarity
            plot_intermediate_feature_L1_similarity_matrix[color_direction_index, contrast_test_index] = L1_similarity_intermediate

            L2_similarity = float(torch.norm(test_feature - reference_feature, p=2).cpu())
            L2_similarity_intermediate = float(
                torch.norm(test_feature_intermediate - reference_feature_intermediate, p=2).cpu())
            csv_data['final_feature_L2_similarity'].append(L2_similarity)
            csv_data['intermediate_feature_L2_similarity'].append(L2_similarity_intermediate)
            plot_final_feature_L2_similarity_matrix[color_direction_index, contrast_test_index] = L2_similarity
            plot_intermediate_feature_L2_similarity_matrix[color_direction_index, contrast_test_index] = L2_similarity_intermediate

            cos_similarity = float(F.cosine_similarity(test_feature, reference_feature).cpu())
            cos_similarity_intermediate = float(F.cosine_similarity(test_feature_intermediate.view(1, -1),
                                                                    reference_feature_intermediate.view(1, -1)).cpu())
            csv_data['final_feature_cos_similarity'].append(cos_similarity)
            csv_data['intermediate_feature_cos_similarity'].append(cos_similarity_intermediate)
            plot_final_feature_cos_similarity_matrix[color_direction_index, contrast_test_index] = cos_similarity
            plot_intermediate_feature_cos_similarity_matrix[color_direction_index, contrast_test_index] = cos_similarity_intermediate

            df = pd.DataFrame(csv_data)
            df.to_csv(os.path.join(save_root_path, f'dinov2_test_on_supra_contrast_color_different_c_contour_plot_ppd_{default_ppd}_temporary.csv'), index=False)
    json_plot_data['color_direction_matrix'].append(plot_color_direction_matrix.tolist())
    json_plot_data['contrast_test_matrix'].append(plot_contrast_test_matrix.tolist())
    json_plot_data['final_feature_L1_similarity_matrix'].append(plot_final_feature_L1_similarity_matrix.tolist())
    json_plot_data['final_feature_L2_similarity_matrix'].append(plot_final_feature_L2_similarity_matrix.tolist())
    json_plot_data['final_feature_cos_similarity_matrix'].append(plot_final_feature_cos_similarity_matrix.tolist())
    json_plot_data['intermediate_feature_L1_similarity_matrix'].append(plot_intermediate_feature_L1_similarity_matrix.tolist())
    json_plot_data['intermediate_feature_L2_similarity_matrix'].append(plot_intermediate_feature_L2_similarity_matrix.tolist())
    json_plot_data['intermediate_feature_cos_similarity_matrix'].append(plot_intermediate_feature_cos_similarity_matrix.tolist())

df = pd.DataFrame(csv_data)
df.to_csv(os.path.join(save_root_path, f'dinov2_test_on_supra_contrast_color_different_c_contour_plot_ppd_{default_ppd}_final.csv'), index=False)
with open(os.path.join(save_root_path, f'dinov2_test_on_supra_contrast_color_different_c_contour_plot_ppd_{default_ppd}_final.json'), 'w') as fp:
    json.dump(json_plot_data, fp)

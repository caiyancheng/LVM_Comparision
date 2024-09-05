import numpy as np
import torch
from Gabor_test_stimulus_generator.generate_plot_gabor_functions import generate_gabor_patch
import torch.nn.functional as F
import pandas as pd
import json
from tqdm import tqdm
import os
torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt

# Only test cpd right now
# Dinov2 input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/dinov2/different_luminance'
os.makedirs(save_root_path, exist_ok=True)
all_backbone_list = ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14',
                     'dinov2_vits14_reg', 'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg']
# all_backbone_list = ['dinov2_vits14']

display_encoded_a = 400
display_encoded_gamma = 2.2

default_W = 224
default_H = 224
default_R = 1
default_rho = 2
contrast_list = np.logspace(np.log10(0.001), np.log10(1), 20)
default_O = 0
default_contrast = 1
C_b_list = np.logspace(np.log10(0.07), np.log10(0.5), 20)
L_b_list = display_encoded_a * C_b_list ** display_encoded_gamma
default_ppd = 60

csv_data = {}
csv_data['backbone_name'] = []
csv_data['C_b'] = []
csv_data['L_b'] = []
csv_data['contrast'] = []
csv_data['final_feature_L1_similarity'] = []
csv_data['final_feature_L2_similarity'] = []
csv_data['final_feature_cos_similarity'] = []
csv_data['intermediate_feature_L1_similarity'] = []
csv_data['intermediate_feature_L2_similarity'] = []
csv_data['intermediate_feature_cos_similarity'] = []

json_plot_data = {}
json_plot_data['backbone_name'] = []
json_plot_data['C_b_matrix'] = []
json_plot_data['L_b_matrix'] = []
json_plot_data['contrast_matrix'] = []
json_plot_data['final_feature_L1_similarity_matrix'] = []
json_plot_data['final_feature_L2_similarity_matrix'] = []
json_plot_data['final_feature_cos_similarity_matrix'] = []
json_plot_data['intermediate_feature_L1_similarity_matrix'] = []
json_plot_data['intermediate_feature_L2_similarity_matrix'] = []
json_plot_data['intermediate_feature_cos_similarity_matrix'] = []

for backbone_name in tqdm(all_backbone_list):
    json_plot_data['backbone_name'].append(backbone_name)
    plot_C_b_matrix = np.zeros([len(C_b_list), len(contrast_list)])
    plot_L_b_matrix = np.zeros([len(C_b_list), len(contrast_list)])
    plot_contrast_matrix = np.zeros([len(C_b_list), len(contrast_list)])
    plot_final_feature_L1_similarity_matrix = np.zeros([len(C_b_list), len(contrast_list)])
    plot_final_feature_L2_similarity_matrix = np.zeros([len(C_b_list), len(contrast_list)])
    plot_final_feature_cos_similarity_matrix = np.zeros([len(C_b_list), len(contrast_list)])
    plot_intermediate_feature_L1_similarity_matrix = np.zeros([len(C_b_list), len(contrast_list)])
    plot_intermediate_feature_L2_similarity_matrix = np.zeros([len(C_b_list), len(contrast_list)])
    plot_intermediate_feature_cos_similarity_matrix = np.zeros([len(C_b_list), len(contrast_list)])

    backbone_model = torch.hub.load('facebookresearch/dinov2', backbone_name)
    backbone_model.eval()
    backbone_model.cuda()

    for C_b_index in range(len(C_b_list)):
        C_b_value = C_b_list[C_b_index]
        L_b_value = display_encoded_a * C_b_value ** display_encoded_gamma
        reference_pattern = C_b_value * torch.ones([1, 3, default_H, default_W])
        reference_feature = backbone_model(reference_pattern.cuda())
        reference_feature_intermediate = backbone_model.get_intermediate_layers(reference_pattern.cuda(), n=4)
        reference_feature_intermediate = torch.stack(reference_feature_intermediate)
        for contrast_index in range(len(contrast_list)):
            contrast_value = contrast_list[contrast_index]
            csv_data['backbone_name'].append(backbone_name)
            csv_data['C_b'].append(C_b_value)
            csv_data['L_b'].append(L_b_value)
            csv_data['contrast'].append(contrast_value)
            plot_C_b_matrix[C_b_index, contrast_index] = C_b_value
            plot_L_b_matrix[C_b_index, contrast_index] = L_b_value
            plot_contrast_matrix[C_b_index, contrast_index] = contrast_value
            gabor_test = generate_gabor_patch(W=default_W, H=default_H, R=default_R, rho=default_rho, O=default_O,
                                              C_b=C_b_value, contrast=contrast_value, ppd=default_ppd)
            gabor_test = torch.tensor(gabor_test, dtype=torch.float32)[None, None, ...] / 255
            gabor_test = gabor_test.expand(-1, 3, -1, -1)
            test_feature = backbone_model(gabor_test.cuda())
            test_feature_intermediate = backbone_model.get_intermediate_layers(gabor_test.cuda(), n=4)
            test_feature_intermediate = torch.stack(test_feature_intermediate)

            L1_similarity = float(torch.norm(test_feature - reference_feature, p=1).cpu())
            L1_similarity_intermediate = float(torch.norm(test_feature_intermediate - reference_feature_intermediate, p=1).cpu())
            csv_data['final_feature_L1_similarity'].append(L1_similarity)
            csv_data['intermediate_feature_L1_similarity'].append(L1_similarity_intermediate)
            plot_final_feature_L1_similarity_matrix[C_b_index, contrast_index] = L1_similarity
            plot_intermediate_feature_L1_similarity_matrix[C_b_index, contrast_index] = L1_similarity_intermediate

            L2_similarity = float(torch.norm(test_feature - reference_feature, p=2).cpu())
            L2_similarity_intermediate = float(torch.norm(test_feature_intermediate - reference_feature_intermediate, p=2).cpu())
            csv_data['final_feature_L2_similarity'].append(L2_similarity)
            csv_data['intermediate_feature_L2_similarity'].append(L2_similarity_intermediate)
            plot_final_feature_L2_similarity_matrix[C_b_index, contrast_index] = L2_similarity
            plot_intermediate_feature_L2_similarity_matrix[C_b_index, contrast_index] = L2_similarity_intermediate

            cos_similarity = float(F.cosine_similarity(test_feature, reference_feature).cpu())
            cos_similarity_intermediate = float(F.cosine_similarity(test_feature_intermediate.view(1,-1), reference_feature_intermediate.view(1,-1)).cpu())
            csv_data['final_feature_cos_similarity'].append(cos_similarity)
            csv_data['intermediate_feature_cos_similarity'].append(cos_similarity_intermediate)
            plot_final_feature_cos_similarity_matrix[C_b_index, contrast_index] = cos_similarity
            plot_intermediate_feature_cos_similarity_matrix[C_b_index, contrast_index] = cos_similarity_intermediate

            df = pd.DataFrame(csv_data)
            df.to_csv(os.path.join(save_root_path, f'dinov2_test_on_gabors_different_C_b_contour_plot_ppd_{default_ppd}_temporary.csv'), index=False)
    json_plot_data['C_b_matrix'].append(plot_C_b_matrix.tolist())
    json_plot_data['L_b_matrix'].append(plot_L_b_matrix.tolist())
    json_plot_data['contrast_matrix'].append(plot_contrast_matrix.tolist())
    json_plot_data['final_feature_L1_similarity_matrix'].append(plot_final_feature_L1_similarity_matrix.tolist())
    json_plot_data['final_feature_L2_similarity_matrix'].append(plot_final_feature_L2_similarity_matrix.tolist())
    json_plot_data['final_feature_cos_similarity_matrix'].append(plot_final_feature_cos_similarity_matrix.tolist())
    json_plot_data['intermediate_feature_L1_similarity_matrix'].append(plot_intermediate_feature_L1_similarity_matrix.tolist())
    json_plot_data['intermediate_feature_L2_similarity_matrix'].append(plot_intermediate_feature_L2_similarity_matrix.tolist())
    json_plot_data['intermediate_feature_cos_similarity_matrix'].append(plot_intermediate_feature_cos_similarity_matrix.tolist())

df = pd.DataFrame(csv_data)
df.to_csv(os.path.join(save_root_path, f'dinov2_test_on_gabors_different_C_b_contour_plot_ppd_{default_ppd}_final.csv'), index=False)
with open(os.path.join(save_root_path, f'dinov2_test_on_gabors_different_C_b_contour_plot_ppd_{default_ppd}_final.json'), 'w') as fp:
    json.dump(json_plot_data, fp)

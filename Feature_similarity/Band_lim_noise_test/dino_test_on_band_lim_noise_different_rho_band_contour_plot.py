import numpy as np
import torch
from Band_limit_noise_generator.generate_plot_band_lim_noise import generate_band_lim_noise
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
# dino input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/dino/different_rho'
os.makedirs(save_root_path, exist_ok=True)
all_backbone_list = ['dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8', 'dino_xcit_small_12_p16',
                     'dino_xcit_small_12_p8', 'dino_xcit_medium_24_p16', 'dino_xcit_medium_24_p8', 'dino_resnet50']

default_W = 224
default_H = 224
rho_list = np.logspace(np.log10(0.5), np.log10(32), 20)
default_L_b = 100
contrast_list = np.logspace(np.log10(0.001), np.log10(1), 20)
default_ppd = 60

csv_data = {}
csv_data['backbone_name'] = []
csv_data['rho'] = []
csv_data['contrast'] = []
csv_data['final_feature_L1_similarity'] = []
csv_data['final_feature_L2_similarity'] = []
csv_data['final_feature_cos_similarity'] = []
csv_data['intermediate_feature_L1_similarity'] = []
csv_data['intermediate_feature_L2_similarity'] = []
csv_data['intermediate_feature_cos_similarity'] = []

json_plot_data = {}
json_plot_data['backbone_name'] = []
json_plot_data['rho_matrix'] = []
json_plot_data['contrast_matrix'] = []
json_plot_data['final_feature_L1_similarity_matrix'] = []
json_plot_data['final_feature_L2_similarity_matrix'] = []
json_plot_data['final_feature_cos_similarity_matrix'] = []
json_plot_data['intermediate_feature_L1_similarity_matrix'] = []
json_plot_data['intermediate_feature_L2_similarity_matrix'] = []
json_plot_data['intermediate_feature_cos_similarity_matrix'] = []

for backbone_name in tqdm(all_backbone_list):
    json_plot_data['backbone_name'].append(backbone_name)
    plot_rho_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_contrast_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_L1_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_L2_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_final_feature_cos_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_intermediate_feature_L1_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_intermediate_feature_L2_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])
    plot_intermediate_feature_cos_similarity_matrix = np.zeros([len(rho_list), len(contrast_list)])

    backbone_model = torch.hub.load('facebookresearch/dino:main', backbone_name)
    backbone_model.eval()
    backbone_model.cuda()

    for rho_index in range(len(rho_list)):
        rho_value = rho_list[rho_index]
        for contrast_index in range(len(contrast_list)):
            contrast_value = contrast_list[contrast_index]
            csv_data['backbone_name'].append(backbone_name)
            csv_data['rho'].append(rho_value)
            csv_data['contrast'].append(contrast_value)
            plot_rho_matrix[rho_index, contrast_index] = rho_value
            plot_contrast_matrix[rho_index, contrast_index] = contrast_value
            T_vid, R_vid = generate_band_lim_noise(W=default_W, H=default_H, freq_band=rho_value,
                                                   L_b=default_L_b, contrast=contrast_value, ppd=default_ppd)
            T_vid_c = display_encode_tool.L2C(T_vid)
            R_vid_c = display_encode_tool.L2C(R_vid)
            T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32)[None, None, ...]
            R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32)[None, None, ...]
            T_vid_ct = T_vid_ct.expand(-1, 3, -1, -1).cuda()
            R_vid_ct = R_vid_ct.expand(-1, 3, -1, -1).cuda()
            test_feature = backbone_model(T_vid_ct)
            reference_feature = backbone_model(R_vid_ct)
            L1_similarity = float(torch.norm(test_feature - reference_feature, p=1).cpu())
            L2_similarity = float(torch.norm(test_feature - reference_feature, p=2).cpu())
            cos_similarity = float(F.cosine_similarity(test_feature, reference_feature).cpu())

            if backbone_name.startswith('dino_vit'):
                test_feature_intermediate = backbone_model.get_intermediate_layers(T_vid_ct, n=4)
                test_feature_intermediate = torch.stack(test_feature_intermediate)
                reference_feature_intermediate = backbone_model.get_intermediate_layers(R_vid_ct, n=4)
                reference_feature_intermediate = torch.stack(reference_feature_intermediate)
                L1_similarity_intermediate = float(
                    torch.norm(test_feature_intermediate - reference_feature_intermediate, p=1).cpu())
                L2_similarity_intermediate = float(
                    torch.norm(test_feature_intermediate - reference_feature_intermediate, p=2).cpu())
                cos_similarity_intermediate = float(F.cosine_similarity(test_feature_intermediate.view(1, -1),
                                                                        reference_feature_intermediate.view(1,
                                                                                                            -1)).cpu())
            else:
                L1_similarity_intermediate = 0
                L2_similarity_intermediate = 0
                cos_similarity_intermediate = 0

            L1_similarity = float(torch.norm(test_feature - reference_feature, p=1).cpu())
            csv_data['final_feature_L1_similarity'].append(L1_similarity)
            csv_data['intermediate_feature_L1_similarity'].append(L1_similarity_intermediate)
            plot_final_feature_L1_similarity_matrix[rho_index, contrast_index] = L1_similarity
            plot_intermediate_feature_L1_similarity_matrix[rho_index, contrast_index] = L1_similarity_intermediate

            L2_similarity = float(torch.norm(test_feature - reference_feature, p=2).cpu())
            csv_data['final_feature_L2_similarity'].append(L2_similarity)
            csv_data['intermediate_feature_L2_similarity'].append(L2_similarity_intermediate)
            plot_final_feature_L2_similarity_matrix[rho_index, contrast_index] = L2_similarity
            plot_intermediate_feature_L2_similarity_matrix[rho_index, contrast_index] = L2_similarity_intermediate

            cos_similarity = float(F.cosine_similarity(test_feature, reference_feature).cpu())
            csv_data['final_feature_cos_similarity'].append(cos_similarity)
            csv_data['intermediate_feature_cos_similarity'].append(cos_similarity_intermediate)
            plot_final_feature_cos_similarity_matrix[rho_index, contrast_index] = cos_similarity
            plot_intermediate_feature_cos_similarity_matrix[rho_index, contrast_index] = cos_similarity_intermediate

            df = pd.DataFrame(csv_data)
            df.to_csv(os.path.join(save_root_path, f'dino_test_on_band_lim_noise_different_rho_band_contour_plot_ppd_{default_ppd}_temporary.csv'), index=False)
    json_plot_data['rho_matrix'].append(plot_rho_matrix.tolist())
    json_plot_data['contrast_matrix'].append(plot_contrast_matrix.tolist())
    json_plot_data['final_feature_L1_similarity_matrix'].append(plot_final_feature_L1_similarity_matrix.tolist())
    json_plot_data['final_feature_L2_similarity_matrix'].append(plot_final_feature_L2_similarity_matrix.tolist())
    json_plot_data['final_feature_cos_similarity_matrix'].append(plot_final_feature_cos_similarity_matrix.tolist())
    json_plot_data['intermediate_feature_L1_similarity_matrix'].append(plot_intermediate_feature_L1_similarity_matrix.tolist())
    json_plot_data['intermediate_feature_L2_similarity_matrix'].append(plot_intermediate_feature_L2_similarity_matrix.tolist())
    json_plot_data['intermediate_feature_cos_similarity_matrix'].append(plot_intermediate_feature_cos_similarity_matrix.tolist())

df = pd.DataFrame(csv_data)
df.to_csv(os.path.join(save_root_path, f'dino_test_on_band_lim_noise_different_rho_band_contour_plot_ppd_{default_ppd}_final.csv'), index=False)
with open(os.path.join(save_root_path, f'dino_test_on_band_lim_noise_different_rho_band_contour_plot_ppd_{default_ppd}_final.json'), 'w') as fp:
    json.dump(json_plot_data, fp)

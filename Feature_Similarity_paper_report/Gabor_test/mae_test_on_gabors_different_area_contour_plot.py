import sys
sys.path.append('E:\Py_codes\LVM_Comparision')
import numpy as np
import torch
from Gabor_test_stimulus_generator.generate_plot_gabor_functions_new import generate_gabor_patch
import torch.nn.functional as F
import pandas as pd
import json
from tqdm import tqdm
import os
import math
from PIL import Image
from transformers import AutoImageProcessor, ViTMAEForPreTraining

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt
from display_encoding import display_encode

display_encode_tool = display_encode(400)

# Only test cpd right now
# Dino input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/mae/different_area'
os.makedirs(save_root_path, exist_ok=True)
all_mae_model_list = ['vit-mae-base', 'vit-mae-large', 'vit-mae-huge']

default_W = 224
default_H = 224
R_min = 0.1
R_max = 1
Area_list = np.logspace(np.log10(math.pi * R_min ** 2), np.log10(math.pi * R_max ** 2), 20)
R_list = (Area_list / math.pi) ** 0.5
# rho_list = [0.5, 1, 2, 4, 8, 16, 32]
default_rho = 8
contrast_list = np.logspace(np.log10(0.001), np.log10(1), 20)
default_O = 0
default_contrast = 1
default_L_b = 100
default_ppd = 60

csv_data = {}
csv_data['mae_model_name'] = []
csv_data['Radius'] = []
csv_data['Area'] = []
csv_data['contrast'] = []
csv_data['final_feature_L1_similarity'] = []
csv_data['final_feature_L2_similarity'] = []
csv_data['final_feature_cos_similarity'] = []

json_plot_data = {}
json_plot_data['mae_model_name'] = []
json_plot_data['radius_matrix'] = []
json_plot_data['area_matrix'] = []
json_plot_data['contrast_matrix'] = []
json_plot_data['final_feature_L1_similarity_matrix'] = []
json_plot_data['final_feature_L2_similarity_matrix'] = []
json_plot_data['final_feature_cos_similarity_matrix'] = []

for mae_model_index in tqdm(range(len(all_mae_model_list))):
    mae_model_name = all_mae_model_list[mae_model_index]
    plot_radius_matrix = np.zeros([len(R_list), len(contrast_list)])
    plot_area_matrix = np.zeros([len(R_list), len(contrast_list)])
    plot_contrast_matrix = np.zeros([len(R_list), len(contrast_list)])
    plot_final_feature_L1_similarity_matrix = np.zeros([len(R_list), len(contrast_list)])
    plot_final_feature_L2_similarity_matrix = np.zeros([len(R_list), len(contrast_list)])
    plot_final_feature_cos_similarity_matrix = np.zeros([len(R_list), len(contrast_list)])

    processor = AutoImageProcessor.from_pretrained(f'facebook/{mae_model_name}')
    model = ViTMAEForPreTraining.from_pretrained(f'facebook/{mae_model_name}')
    model.eval()

    for R_index in range(len(R_list)):
        R_value = R_list[R_index]
        A_value = Area_list[R_index]
        for contrast_index in range(len(contrast_list)):
            contrast_value = contrast_list[contrast_index]
            csv_data['mae_model_name'].append(mae_model_name)
            csv_data['Radius'].append(R_value)
            csv_data['Area'].append(A_value)
            csv_data['contrast'].append(contrast_value)
            plot_radius_matrix[R_index, contrast_index] = R_value
            plot_area_matrix[R_index, contrast_index] = A_value
            plot_contrast_matrix[R_index, contrast_index] = contrast_value
            T_vid, R_vid = generate_gabor_patch(W=default_W, H=default_H, R=R_value, rho=default_rho, O=default_O,
                                                L_b=default_L_b, contrast=contrast_value, ppd=default_ppd,
                                                color_direction='ach')
            T_vid_c = Image.fromarray((display_encode_tool.L2C_sRGB(T_vid) * 255).astype(np.uint8))
            R_vid_c = Image.fromarray((display_encode_tool.L2C_sRGB(R_vid) * 255).astype(np.uint8))
            T_vid_ct = processor(images=T_vid_c, return_tensors="pt")
            R_vid_ct = processor(images=R_vid_c, return_tensors="pt")
            T_output = model(**T_vid_ct)
            R_output = model(**R_vid_ct)
            T_features = T_output.logits
            R_features = R_output.logits
            L1_similarity = float(torch.norm(T_features - R_features, p=1).cpu())
            L2_similarity = float(torch.norm(T_features - R_features, p=2).cpu())
            cos_similarity = float(F.cosine_similarity(T_features.view(1, -1), R_features.view(1, -1)).cpu())

            csv_data['final_feature_L1_similarity'].append(L1_similarity)
            plot_final_feature_L1_similarity_matrix[R_index, contrast_index] = L1_similarity

            csv_data['final_feature_L2_similarity'].append(L2_similarity)
            plot_final_feature_L2_similarity_matrix[R_index, contrast_index] = L2_similarity

            csv_data['final_feature_cos_similarity'].append(cos_similarity)
            plot_final_feature_cos_similarity_matrix[R_index, contrast_index] = cos_similarity

            df = pd.DataFrame(csv_data)
            df.to_csv(os.path.join(save_root_path,
                                   f'mae_test_on_gabors_different_area_contour_plot_ppd_{default_ppd}_temporary.csv'),
                      index=False)
    json_plot_data['mae_model_name'].append(mae_model_name)
    json_plot_data['radius_matrix'].append(plot_radius_matrix.tolist())
    json_plot_data['area_matrix'].append(plot_area_matrix.tolist())
    json_plot_data['contrast_matrix'].append(plot_contrast_matrix.tolist())
    json_plot_data['final_feature_L1_similarity_matrix'].append(plot_final_feature_L1_similarity_matrix.tolist())
    json_plot_data['final_feature_L2_similarity_matrix'].append(plot_final_feature_L2_similarity_matrix.tolist())
    json_plot_data['final_feature_cos_similarity_matrix'].append(plot_final_feature_cos_similarity_matrix.tolist())

df = pd.DataFrame(csv_data)
df.to_csv(
    os.path.join(save_root_path, f'mae_test_on_gabors_different_area_contour_plot_ppd_{default_ppd}_final.csv'),
    index=False)
with open(
        os.path.join(save_root_path,
                     f'mae_test_on_gabors_different_area_contour_plot_ppd_{default_ppd}_final.json'),
        'w') as fp:
    json.dump(json_plot_data, fp)

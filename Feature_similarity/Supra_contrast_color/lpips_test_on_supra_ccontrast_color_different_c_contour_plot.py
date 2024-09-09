import numpy as np
import torch
from Supra_contrast_color_generator.generate_plot_supra_contrast_color import generate_supra_contrast_color
import torch.nn.functional as F
import pandas as pd
import json
from tqdm import tqdm
import lpips
import os
from display_encoding import display_encode
display_encode_tool = display_encode(400, 2.2)

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt

# Only test cpd right now
# input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/lpips/different_c'
os.makedirs(save_root_path, exist_ok=True)
loss_fn_alex = lpips.LPIPS(net='alex').eval()
loss_fn_vgg = lpips.LPIPS(net='vgg').eval()
loss_fn_squeeze = lpips.LPIPS(net='squeeze').eval()

default_W = 224
default_H = 224
default_rho = 1
default_L_b = 100
contrast_test_list = np.logspace(np.log10(0.01), np.log10(0.2), 20)
default_ppd = 60
color_direction_list = [0, 1, 2] #['ach', 'rg', 'yv']
color_direction_map = ['ach', 'rg', 'yv']

csv_data = {}
csv_data['contrast_test'] = []
csv_data['color_direction'] = []
csv_data['loss_fn_alex'] = []
csv_data['loss_fn_vgg'] = []
csv_data['loss_fn_squeeze'] = []

json_plot_data = {}
json_plot_data['contrast_test_matrix'] = []
json_plot_data['color_direction_matrix'] = []
json_plot_data['loss_fn_alex_matrix'] = []
json_plot_data['loss_fn_vgg_matrix'] = []
json_plot_data['loss_fn_squeeze_matrix'] = []

plot_contrast_test_matrix = np.zeros([len(color_direction_list), len(contrast_test_list)])
plot_color_direction_matrix = np.zeros([len(color_direction_list), len(contrast_test_list)])
plot_loss_fn_alex_matrix = np.zeros([len(color_direction_list), len(contrast_test_list)])
plot_loss_fn_vgg_matrix = np.zeros([len(color_direction_list), len(contrast_test_list)])
plot_loss_fn_squeeze_matrix = np.zeros([len(color_direction_list), len(contrast_test_list)])

for contrast_test_index in range(len(contrast_test_list)):
    contrast_test_value = contrast_test_list[contrast_test_index]
    for color_direction_index in range(len(color_direction_list)):
        color_direction_value = color_direction_list[color_direction_index]
        csv_data['contrast_test'].append(contrast_test_value)
        csv_data['color_direction'].append(color_direction_value)
        plot_contrast_test_matrix[color_direction_index, contrast_test_index] = contrast_test_value
        plot_color_direction_matrix[color_direction_index, contrast_test_index] = color_direction_value
        T_vid, R_vid = generate_supra_contrast_color(W=default_W, H=default_H, rho=default_rho, L_b=default_L_b,
                                                     contrast_test=contrast_test_value, ppd=default_ppd,
                                                     color_direction=color_direction_map[color_direction_value])
        T_vid_c = display_encode_tool.L2C(T_vid)
        R_vid_c = display_encode_tool.L2C(R_vid)
        T_vid_ct = torch.tensor(T_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...]
        R_vid_ct = torch.tensor(R_vid_c, dtype=torch.float32).permute(2, 0, 1)[None, ...]
        norm_T_vid_ct = (T_vid_ct - 0.5) * 2
        norm_R_vid_ct = (R_vid_ct - 0.5) * 2
        loss_fn_alex_value = float(loss_fn_alex(norm_T_vid_ct, norm_R_vid_ct).cpu())
        loss_fn_vgg_value = float(loss_fn_vgg(norm_T_vid_ct, norm_R_vid_ct).cpu())
        loss_fn_squeeze_value = float(loss_fn_squeeze(norm_T_vid_ct, norm_R_vid_ct).cpu())

        csv_data['loss_fn_alex'].append(loss_fn_alex_value)
        plot_loss_fn_alex_matrix[color_direction_index, contrast_test_index] = loss_fn_alex_value
        csv_data['loss_fn_vgg'].append(loss_fn_vgg_value)
        plot_loss_fn_vgg_matrix[color_direction_index, contrast_test_index] = loss_fn_vgg_value
        csv_data['loss_fn_squeeze'].append(loss_fn_squeeze_value)
        plot_loss_fn_squeeze_matrix[color_direction_index, contrast_test_index] = loss_fn_squeeze_value

        df = pd.DataFrame(csv_data)
        df.to_csv(os.path.join(save_root_path, f'lpips_test_on_supra_contrast_color_different_c_contour_plot_ppd_{default_ppd}_temporary.csv'),
                  index=False)
json_plot_data['color_direction_matrix'].append(plot_color_direction_matrix.tolist())
json_plot_data['contrast_test_matrix'].append(plot_contrast_test_matrix.tolist())
json_plot_data['loss_fn_alex_matrix'].append(plot_loss_fn_alex_matrix.tolist())
json_plot_data['loss_fn_vgg_matrix'].append(plot_loss_fn_vgg_matrix.tolist())
json_plot_data['loss_fn_squeeze_matrix'].append(plot_loss_fn_squeeze_matrix.tolist())

df = pd.DataFrame(csv_data)
df.to_csv(os.path.join(save_root_path, f'lpips_test_on_supra_contrast_color_different_c_contour_plot_ppd_{default_ppd}_final.csv'), index=False)
with open(os.path.join(save_root_path, f'lpips_test_on_supra_contrast_color_different_c_contour_plot_ppd_{default_ppd}_final.json'), 'w') as fp:
    json.dump(json_plot_data, fp)

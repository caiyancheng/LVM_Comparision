import numpy as np
import torch
from Gabor_test_stimulus_generator.generate_plot_gabor_functions import generate_gabor_patch
import torch.nn.functional as F
import pandas as pd
import json
from tqdm import tqdm
import lpips
import os

torch.hub.set_dir(r'E:\Torch_hub')
import matplotlib.pyplot as plt

# Only test cpd right now
# input: Batch, Channel, H, W // Value = [0,1]
save_root_path = 'new_data_logs/lpips/different_luminance'
os.makedirs(save_root_path, exist_ok=True)

loss_fn_alex = lpips.LPIPS(net='alex').eval()
loss_fn_vgg = lpips.LPIPS(net='vgg').eval()
loss_fn_squeeze = lpips.LPIPS(net='squeeze').eval()

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
csv_data['C_b'] = []
csv_data['L_b'] = []
csv_data['contrast'] = []
csv_data['loss_fn_alex'] = []
csv_data['loss_fn_vgg'] = []
csv_data['loss_fn_squeeze'] = []

json_plot_data = {}
json_plot_data['C_b_matrix'] = []
json_plot_data['L_b_matrix'] = []
json_plot_data['contrast_matrix'] = []
json_plot_data['loss_fn_alex_matrix'] = []
json_plot_data['loss_fn_vgg_matrix'] = []
json_plot_data['loss_fn_squeeze_matrix'] = []



plot_C_b_matrix = np.zeros([len(C_b_list), len(contrast_list)])
plot_L_b_matrix = np.zeros([len(C_b_list), len(contrast_list)])
plot_contrast_matrix = np.zeros([len(C_b_list), len(contrast_list)])
plot_loss_fn_alex_matrix = np.zeros([len(C_b_list), len(contrast_list)])
plot_loss_fn_vgg_matrix = np.zeros([len(C_b_list), len(contrast_list)])
plot_loss_fn_squeeze_matrix = np.zeros([len(C_b_list), len(contrast_list)])

for C_b_index in range(len(C_b_list)):
    C_b_value = C_b_list[C_b_index]
    L_b_value = display_encoded_a * C_b_value ** display_encoded_gamma
    reference_pattern = C_b_value * torch.ones([1, 3, default_H, default_W])
    norm_reference_pattern = (reference_pattern - 0.5) * 2
    for contrast_index in range(len(contrast_list)):
        contrast_value = contrast_list[contrast_index]
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
        norm_gabor_test = (gabor_test - 0.5) * 2

        loss_fn_alex_value = float(loss_fn_alex(norm_reference_pattern, norm_gabor_test).cpu())
        loss_fn_vgg_value = float(loss_fn_vgg(norm_reference_pattern, norm_gabor_test).cpu())
        loss_fn_squeeze_value = float(loss_fn_squeeze(norm_reference_pattern, norm_gabor_test).cpu())

        csv_data['loss_fn_alex'].append(loss_fn_alex_value)
        plot_loss_fn_alex_matrix[C_b_index, contrast_index] = loss_fn_alex_value
        csv_data['loss_fn_vgg'].append(loss_fn_vgg_value)
        plot_loss_fn_vgg_matrix[C_b_index, contrast_index] = loss_fn_vgg_value
        csv_data['loss_fn_squeeze'].append(loss_fn_squeeze_value)
        plot_loss_fn_squeeze_matrix[C_b_index, contrast_index] = loss_fn_squeeze_value

        df = pd.DataFrame(csv_data)
        df.to_csv(os.path.join(save_root_path, f'lpips_test_on_gabors_different_L_b_contour_plot_ppd_{default_ppd}_temporary.csv'), index=False)
json_plot_data['C_b_matrix'].append(plot_C_b_matrix.tolist())
json_plot_data['L_b_matrix'].append(plot_L_b_matrix.tolist())
json_plot_data['contrast_matrix'].append(plot_contrast_matrix.tolist())
json_plot_data['loss_fn_alex_matrix'].append(plot_loss_fn_alex_matrix.tolist())
json_plot_data['loss_fn_vgg_matrix'].append(plot_loss_fn_vgg_matrix.tolist())
json_plot_data['loss_fn_squeeze_matrix'].append(plot_loss_fn_squeeze_matrix.tolist())

df = pd.DataFrame(csv_data)
df.to_csv(os.path.join(save_root_path, f'lpips_test_on_gabors_different_L_b_contour_plot_ppd_{default_ppd}_final.csv'), index=False)
with open(os.path.join(save_root_path, f'lpips_test_on_gabors_different_L_b_contour_plot_ppd_{default_ppd}_final.json'), 'w') as fp:
    json.dump(json_plot_data, fp)

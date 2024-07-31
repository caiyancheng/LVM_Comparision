from generate_plot_gabor_functions import generate_gabor_patch
import numpy as np
import json
from tqdm import tqdm
import os

dataset_root_path = r'E:\Py_codes\LVM_Comparision/Gabor_Dataset_1'
os.makedirs(dataset_root_path, exist_ok=True)

W = 224  # Width of the canvas (pixels)
H = 224  # Height of the canvas (pixels)
R_list = [0.5, 0.75, 1, 1.25, 1.5]  # Radius of the Gabor stimulus (degrees)
rho_list = [0.5, 1, 2, 4, 8, 16]  # Spatial frequency of the Gabor stimulus (cycles per degree)
O = 0  # Orientation of the Gabor stimulus (degrees)
L_b_list = [0.1, 0.2, 0.5]  # Luminance of the background
contrast_list = [0.1, 0.2, 0.5, 1]  # Contrast of the gabor

setting_list = []
for R in R_list:
    for rho in rho_list:
        for L_b in L_b_list:
            for contrast in contrast_list:
                setting = [W, H, R, rho, O, L_b, contrast]
                setting_list.append(setting)

for setting in tqdm(setting_list):
    W, H, R, rho, O, L_b, contrast = setting
    gabor_image =
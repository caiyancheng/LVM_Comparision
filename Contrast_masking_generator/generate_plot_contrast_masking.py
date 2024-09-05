import numpy as np
import matplotlib.pyplot as plt
from display_encoding import display_encode

Luminance_min = 1e-4
display_encode_tool = display_encode(400, 2.2)

def generate_masking_stimulus(W, H, rho, O, L_b, contrast_mask, contrast_test, ppd):
    size_deg = np.array([W, H]) / ppd
    sigma = 2 / rho
    XX, YY = np.meshgrid(np.linspace(0, size_deg[0], W),
                         np.linspace(0, size_deg[1], H))
    gauss_env = np.exp(-((XX - size_deg[0] / 2) ** 2 + (YY - size_deg[1] / 2) ** 2) / (2 * sigma ** 2))
    DD = np.sqrt(XX ** 2 + YY ** 2)
    cosB = (XX * (-np.sin(np.radians(O))) + YY * np.cos(np.radians(O))) / DD
    cosB[0, 0] = 0  # Avoid division by zero
    d = np.sqrt(1 - cosB ** 2) * DD
    img_mask = np.cos(2 * np.pi * d * rho) * contrast_mask * L_b + L_b
    d = XX
    img_target = np.cos(2 * np.pi * d * rho) * contrast_test * L_b * gauss_env
    S = img_mask + img_target
    return S


def generate_contrast_masking(W, H, rho, O, L_b, contrast_mask, contrast_test, ppd):
    T_vid = generate_masking_stimulus(W, H, rho, O, L_b, contrast_mask, contrast_test, ppd)
    R_vid = generate_masking_stimulus(W, H, rho, O, L_b, contrast_mask, 0, ppd)
    gabor[gabor < Luminance_min] = Luminance_min
    return gabor

def plot_gabor(gabor):
    """
    Plot the Gabor stimulus.

    Parameters:
    gabor (numpy.ndarray): The Gabor stimulus to plot
    """
    plt.figure(figsize=(4,4))
    plt.imshow(gabor, cmap='gray', vmin=0, vmax=255, extent=(-W // 2, W // 2, -H // 2, H // 2))
    plt.title(f'Radius = {R} degree, \n S_freq = {rho} cpd, Contrast = {contrast}, \n ppd = {ppd}, W = {W}, H = {H}')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    scale_k1 = 1
    scale_k2 = 1

    # 示例参数
    W = 224 * scale_k2  # Width of the canvas (pixels)
    H = 224 * scale_k2  # Height of the canvas (pixels)
    R = 0.1 * scale_k1 * scale_k2  # Radius of the Gabor stimulus (degrees)
    rho = 2 / scale_k1 / scale_k2  # Spatial frequency of the Gabor stimulus (cycles per degree)
    O = 0  # Orientation of the Gabor stimulus (degrees)
    C_b = 0.5  # Luminance of the background
    contrast = 1  # Contrast of the gabor
    ppd = 60 / scale_k1

    gabor_stimulus = generate_gabor_patch(W, H, R, rho, O, C_b, contrast, ppd)
    plot_gabor(gabor_stimulus)


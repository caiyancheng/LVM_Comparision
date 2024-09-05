import numpy as np
import matplotlib.pyplot as plt

Luminance_min = 1e-4


def generate_gabor_patch(W, H, R, rho, O, L_b, contrast, ppd):
    x = np.linspace(-W // 2, W // 2, W)
    y = np.linspace(-H // 2, H // 2, H)
    X, Y = np.meshgrid(x, y)

    theta = np.deg2rad(O)

    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)

    gaussian = np.exp(-0.5 * (X_rot ** 2 + Y_rot ** 2) / (ppd * R) ** 2)
    sinusoid = np.sin(2 * np.pi * rho * X_rot / ppd) * contrast * C_b
    T_vid = gaussian * sinusoid + C_b
    R_vid = np.ones([W, H]) * L_b

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
    O = 45  # Orientation of the Gabor stimulus (degrees)
    C_b = 0.5  # Luminance of the background
    contrast = 1  # Contrast of the gabor
    ppd = 60 / scale_k1

    gabor_stimulus = generate_gabor_patch(W, H, R, rho, O, C_b, contrast, ppd)
    plot_gabor(gabor_stimulus)


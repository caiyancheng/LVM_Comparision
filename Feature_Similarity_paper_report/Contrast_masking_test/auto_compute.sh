#!/bin/bash

conda activate dinov2_3

python dino_test_on_contrast_masking_different_c_contour_plot.py
python dinov2_test_on_contrast_masking_different_c_contour_plot.py
python lpips_test_on_contrast_masking_different_c_contour_plot.py
python openclip_test_on_contrast_masking_different_c_contour_plot.py
python stlpips_test_on_contrast_masking_different_c_contour_plot.py
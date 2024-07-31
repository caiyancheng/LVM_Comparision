import torch
torch.hub.set_dir(r'E:\Torch_hub')

dinov2_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')




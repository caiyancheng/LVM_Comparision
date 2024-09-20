from transformers import AutoImageProcessor, ViTMAEForPreTraining
from PIL import Image
import requests
import numpy as np

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw) #[480,640,3]

processor = AutoImageProcessor.from_pretrained('facebook/vit-mae-large',)
model = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-large')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
loss = outputs.loss
mask = outputs.mask
feature = outputs.logits # torch.Size([1, 196, 768])
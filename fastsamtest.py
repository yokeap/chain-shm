python -c "
from ultralytics import FastSAM
import cv2
import numpy as np

model = FastSAM('FastSAM-s.pt')  # download ~23MB ครั้งแรก
img = cv2.imread('chain.png')
print('Image shape:', img.shape)

results = model(img, device='cpu', retina_masks=True, imgsz=640, conf=0.4, iou=0.9)

if results[0].masks is not None:
    masks = results[0].masks.data
    print(f'Masks found: {len(masks)}')
    for i, m in enumerate(masks):
        area = m.cpu().numpy().sum()
        print(f'  Mask {i}: area={area:.0f} px')
else:
    print('No masks found')
"

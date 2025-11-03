import os
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as T
from config import DEVICE
from pdf2image import convert_from_path

table_transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
])

def extract_table(file_path, tablenet_model):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in ['.csv', '.xlsx', '.xls']:
        try:
            return pd.read_csv(file_path) if ext == '.csv' else pd.read_excel(file_path)
        except:
            return None

    elif ext in ['.png', '.jpg', '.jpeg', '.pdf']:
        try:
            if ext == '.pdf':
                image = convert_from_path(file_path, dpi=300)[0]
            else:
                image = Image.open(file_path).convert("RGB")

            w, h = image.size
            inp = table_transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                table_mask, _ = tablenet_model(inp)

            mask = (table_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
            mask = cv2.resize(mask, (w, h))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            tables = []
            for cnt in contours:
                x, y, wc, hc = cv2.boundingRect(cnt)
                if wc > 50 and hc > 50:
                    tables.append(image.crop((x, y, x + wc, y + hc)))

            return tables
        except:
            return None

    return None
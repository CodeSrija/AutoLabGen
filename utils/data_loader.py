import os
from torch.utils.data import Dataset
from PIL import Image

def resize_and_pad(img, size=128):
    w, h = img.size
    scale = size / max(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    new_img = Image.new("RGB", (size, size), (255, 255, 255))
    new_img.paste(img, ((size - img.size[0]) // 2, (size - img.size[1]) // 2))
    return new_img

class HandwritingDataset(Dataset):
    def __init__(self, df, img_dir, size=128):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(os.path.join(self.img_dir, row["FILENAME"])).convert("RGB")
        image = resize_and_pad(image, self.size)
        return {"image": image, "text": str(row["IDENTITY"])}
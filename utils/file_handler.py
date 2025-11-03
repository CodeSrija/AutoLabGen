import os
import shutil
from fastapi import UploadFile
from config import LAB_MANUAL_DIR, DATASET_DIR, IMAGE_DIR, ALLOWED_LAB_MANUALS, ALLOWED_DATASETS, ALLOWED_IMAGES

async def save_file(file: UploadFile):
    """Save uploaded file into correct directory."""
    ext = file.filename.split(".")[-1].lower()

    if ext in ALLOWED_LAB_MANUALS:
        folder = LAB_MANUAL_DIR
    elif ext in ALLOWED_DATASETS:
        folder = DATASET_DIR
    elif ext in ALLOWED_IMAGES:
        folder = IMAGE_DIR
    else:
        raise ValueError(f"Unsupported file format: .{ext}")

    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, file.filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return save_path

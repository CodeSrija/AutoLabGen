import os
import fitz
from PIL import Image
import docx2txt
import torch

def pdf_to_images(pdf_path, output_folder="temp_pages"):
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=300)
        img_path = os.path.join(output_folder, f"page_{i + 1}.png")
        pix.save(img_path)
        image_paths.append(img_path)
    return image_paths

def extract_text_from_image(image_path, processor, model):
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return extracted_text.strip()
    except Exception as e:
        print(f"Error extracting text from {image_path}: {e}")
        return ""

def extract_text(file_path, processor, model):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        print("Converting PDF to images...")
        image_files = pdf_to_images(file_path)
        text = ""
        for img in image_files:
            text += extract_text_from_image(img, processor, model) + "\n"
        return text

    elif ext in [".png", ".jpg", ".jpeg"]:
        return extract_text_from_image(file_path, processor, model)

    elif ext in [".docx", ".doc"]:
        # Option 1: Direct text extraction using docx2txt
        return docx2txt.process(file_path)

    else:
        raise ValueError(f"Unsupported file type: {ext}")
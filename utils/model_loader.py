import os
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoImageProcessor, AutoModelForObjectDetection
from config import TROCR_MODEL_PATH, TABLENET_MODEL_PATH

def load_trocr_model():
    try:
        processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_PATH)
        model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_PATH)

        # Configure model
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.eos_token_id = processor.tokenizer.eos_token_id
        model.config.max_length = 64
        model.config.vocab_size = model.config.decoder.vocab_size

        return processor, model

    except Exception as e:
        print(f"Error loading TrOCR model: {e}")
        return None, None

def load_tablenet_model():
    """
    Loads the TableNet/Table Transformer model from local directory.
    """
    try:
        processor = AutoImageProcessor.from_pretrained(TABLENET_MODEL_PATH)
        model = AutoModelForObjectDetection.from_pretrained(TABLENET_MODEL_PATH)
        return processor, model
    except Exception as e:
        print(f"Error loading table extraction model: {e}")
        return None, None
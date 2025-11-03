import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from utils.data_loader import HandwritingDataset

# 1. Paths
BASE = "models/datasets"
train_csv = f"{BASE}/train.csv"
val_csv   = f"{BASE}/val.csv"
train_dir = f"{BASE}/train_images"
val_dir   = f"{BASE}/val_images"

# 2. Load and shrink dataset
train_df = pd.read_csv(train_csv).sample(frac=0.2, random_state=42)
val_df   = pd.read_csv(val_csv).sample(frac=0.2, random_state=42)

train_dataset = HandwritingDataset(train_df, train_dir)
val_dataset   = HandwritingDataset(val_df, val_dir)

# 3. Load base model
MODEL_PATH = "models/trocr"
processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.eos_token_id
model.to("cuda" if torch.cuda.is_available() else "cpu")

# 4. Collator
def collate_batch(batch):
    images = [b["image"] for b in batch]
    texts = [b["text"] for b in batch]
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    labels = processor.tokenizer(texts, padding=True, return_tensors="pt").input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return {"pixel_values": pixel_values, "labels": labels}

# 5. Training
args = Seq2SeqTrainingArguments(
    output_dir="models/trocr_finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=50,
    eval_steps=200,
    save_steps=500,
    learning_rate=5e-5,
    fp16=True,
    report_to="none",
    remove_unused_columns=False
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_batch
)

trainer.train()
model.save_pretrained("models/trocr_finetuned")
processor.save_pretrained("models/trocr_finetuned")
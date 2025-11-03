import os
from dotenv import load_dotenv
import torch 

# Load environment variables from .env file if available
load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Directory Paths ==========
BASE_DIR = "uploads"

LAB_MANUAL_DIR = os.path.join(BASE_DIR, "lab_manuals")
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_DIR = "outputs/lab_reports"

# Ensure required directories exist
for folder in [LAB_MANUAL_DIR, DATASET_DIR, IMAGE_DIR, OUTPUT_DIR]:
    os.makedirs(folder, exist_ok=True)

# ========== Model Paths ==========
TROCR_MODEL_PATH = "models/trocr"
TABLENET_MODEL_PATH = "models/tablenet/"

# ========== Allowed File Types ==========
ALLOWED_LAB_MANUALS = ["pdf", "txt", "md", "docx"]
ALLOWED_DATASETS = ["csv", "xlsx", "xls", "json", "txt", "pdf"]
ALLOWED_IMAGES = ["jpg", "jpeg", "png"]

# ========== API Keys / LLM Models ==========
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)  # Set in .env file
GROQ_MODEL_NAME = "meta-llama/llama-4-maverick-17b-128e-instruct"

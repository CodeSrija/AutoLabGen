import os
from utils.text_extractor import extract_text
from utils.table_extractor import extract_table
from utils.groq_client import generate_from_groq
from utils.model_loader import load_trocr_model, load_tablenet_model

trocr_processor, trocr_model = load_trocr_model()
tablenet_processor, tablenet_model = load_tablenet_model()

async def generate_lab_report(lab_path, dataset_path, user_prompt):
    print("Inside generate_lab_report")

    if not os.path.exists(lab_path):
        print("Lab manual not found:", lab_path)
        return "Lab manual file missing."

    print("Extracting text from:", lab_path)

    try:
        manual_text = extract_text(lab_path, trocr_processor, trocr_model)
        print(manual_text)
        print("Text extracted successfully")
    except Exception as e:
        print("Error extracting text:", e)
        manual_text = ""

    table_data = ""
    if dataset_path:
        print("Extracting tables from:", dataset_path)
        try:
            table_data = extract_table(dataset_path, tablenet_model)
            print("Table extraction complete")
        except Exception as e:
            print("Error extracting tables:", e)

    if not manual_text and not table_data:
        print("No extracted data from either file")
        return "No data extracted. Check your files."

    print("Sending extracted data to Groq API")
    try:
        prompt = f"Lab Manual:\n{manual_text}"
        response = generate_from_groq(prompt)
        print("Groq API returned a response")
        print(response)
        return response
    except Exception as e:
        print("Groq API error:", e)
        return "Groq API failed."

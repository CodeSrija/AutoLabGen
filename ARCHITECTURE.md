## 1. Overview

AutoLabGen is an AI-driven system designed to automate the generation of laboratory reports from lab manuals, datasets, images, and optional user instructions. It integrates document processing, OCR, table extraction, prompt engineering, and LLM-based generation into a unified pipeline.

## 2. System Workflow

1. **Upload Files** → via web interface (FastAPI + HTML form).
2. **Save Files** → stored in `uploads/`.
3. **Extract Text** → `text_extractor.py` uses pdfplumber/docx2txt.
4. **Extract Tables** → `table_extractor.py` using TableNet.
5. **Handwritten Values** → TrOCR (fine-tuned) extracts text from lab image readings.
6. **Create Prompt** → combines extracted text, table data, and user instructions.
7. **Generate Report** → Sent to Groq LLM → returns final lab report.
8. **Show Output** → Displayed on webpage.

## 3. AI Models

| Component         | Model                    | Purpose                                                      |
| ----------------- | ------------------------ | ------------------------------------------------------------ |
| Handwriting OCR   | Fine-tuned TrOCR         | Extracts handwritten values/observations accurately          |
| Table Detection   | TableNet                 | Detects and extracts structured tables from documents/images |
| Report Generation | Groq LLM (Mixtral/LLaMA) | Generates structured lab reports using extracted content     |

## 4. Why Fine-Tuning TrOCR?

* Lab manuals often contain handwritten readings and experimental results.
* General OCR models produce high error rates for cursive or lab-style handwriting.
* Fine-tuning improves:

  * Character recognition accuracy
  * Table correctness in final reports
  * Consistency of numeric observations

## 5. Data Flow Diagram
```
User Uploads (manual + dataset + images + prompts)
        ↓
1. File Handler  
   - Validates files (pdf/docx/img/csv)  
   - Saves to /uploads  
        ↓
2. Content Extraction  
   - Text from PDF/DOCX → TrOCR (fine-tuned)  
   - Tables from PDFs → TableNet  
   - Images stored for report embedding  
        ↓
3. Prompt Builder  
   - Converts extracted content → structured prompt:  
     {Aim, Materials, Procedure, Observations, Dataset Summary}  
        ↓
4. Groq LLM  
   - Generates final lab report  
        ↓
5. Output  
   - .docx / .pdf report  
   - Interaction logs saved
```

---

## Short summary
AutoLabGen automates the manual task of producing lab reports from lab manuals, datasets, images, and an optional user prompt. The pipeline extracts text and tabular data, composes an engineered prompt, and generates a structured lab report via an LLM.

## Project structure
```
project/
├── main.py
├── utils/
│   ├── data_loader.py
│   ├── file_handler.py
│   ├── text_extractor.py
│   ├── table_extractor.py
│   ├── model_loader.py
│   ├── groq_client.py
│   └── report_generator.py
├── models/
│   ├── trocr/
│   ├── tablenet/
│   ├── finetune_trocr.py
│   └── datasets/               # Add data here
├── uploads/
│   ├── lab_manuals/
│   ├── datasets/
│   └── images/
├── frontend/
│   ├── index.html
│   └── styles.css
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
├── DATA_SCIENCE_REPORT.md
├── INTERACTION_LOGS.md
└── RUN_INSTRUCTIONS.md
```
## Quick start (local)
1. Create environment and install deps:

    ```bash
    python -m venv .venv
    # linux/mac
    source .venv/bin/activate
    # windows
    .venv\Scripts\activate
    pip install -r requirements.txt
    ```

2. Create `.env` with your key:

    ```bash
    GROQ_API_KEY=your_real_api_key_here
    ```

3. Download models & (optional) fine-tune TrOCR
    - Clone baseline models

        ```bash
        git clone https://huggingface.co/microsoft/trocr-base-handwritten models/trocr
        git clone https://huggingface.co/<tablenet-repo> models/tablenet
        ```

    - Fine-tune TrOCR:

        ```bash
        python models/finetune_trocr.py
        ```

4. Start server:

    ```bash
    uvicorn main:app --reload
    ```
    
5. Open `http://127.0.0.1:8000`.

You just need to add a small section in your `README.md`. Keep it simple and professional. Add this at the top or bottom:

---

* **Name:** Duruseti Srija
* **University:** Indian Institute of Technology (IIT) Kanpur
* **Department:** Mechanical Engineering

---

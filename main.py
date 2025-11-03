from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from utils.file_handler import save_file
from utils.report_generator import generate_lab_report

import warnings
warnings.filterwarnings("ignore")

app = FastAPI()
templates = Jinja2Templates(directory="frontend")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "report": None})

@app.post("/generate_report", response_class=HTMLResponse)
async def generate_report(
    request: Request,
    lab_manual: UploadFile = File(...),
    dataset: UploadFile = File(None),
    prompt: str = Form("")
):
    lab_path = await save_file(lab_manual)
    dataset_path = await save_file(dataset) if dataset else None

    report_text = await generate_lab_report(lab_path, dataset_path, prompt)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "report": report_text
    })

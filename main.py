from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from paddleocr import PaddleOCR
from pykospacing import Spacing
from transformers import T5ForConditionalGeneration, T5Tokenizer
from googletrans import Translator

from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Jinja2 템플릿 설정
templates = Jinja2Templates(directory="templates")

# 정적 파일 디렉토리를 지정하고 FastAPI 앱에 마운트
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML 렌더링 엔드포인트
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, translated_text: str = ""):
    return templates.TemplateResponse("index.html", {"request": request, "translated_text": translated_text})


# 모델 로드
model_name = "j5ng/et5-typos-corrector"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
ocr = PaddleOCR(lang="korean")  # OCR 엔진 초기화 (한국어 설정)

def correct_spacing(text):
    spacing = Spacing()
    return spacing(text)

def correct_typo(text, model, tokenizer):
    inputs = tokenizer.encode("correct: " + text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def translate_text(text, dest_language='en_XX'):
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    article_eng = text
    # tokenizer.src_lang = "ko_KR"
    encoded_eng = tokenizer(article_eng, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded_eng,
        forced_bos_token_id=tokenizer.lang_code_to_id[dest_language]
    )
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return result

def read_imagefile(file) -> np.ndarray:
    image = cv2.imdecode(np.fromstring(file, np.uint8), 1)
    return image

import os

@app.post("/process-image/")
async def process_image(request: Request, file: UploadFile = File(...), target_lang: str = Form(...)):
    # 이미지 데이터 읽기
    image_data = await file.read()

    # 이미지 파일 저장
    save_path = "static/images"
    os.makedirs(save_path, exist_ok=True)
    image_save_path = os.path.join(save_path, "temp.jpg")
    with open(image_save_path, "wb") as f:
        f.write(image_data)

    # OCR 처리 및 번역
    result = ocr.ocr(image_save_path, cls=False)
    extracted_text = " ".join([line[1][0] for line in result[0]])
    corrected_text = correct_spacing(extracted_text)
    final_text = correct_typo(corrected_text, model, tokenizer)
    translated_text = translate_text(final_text, dest_language=target_lang)

    # HTML 템플릿에 번역된 텍스트 전달
    return templates.TemplateResponse(
        "index.html",
        {"translated_text": translated_text[0], "request": request},
        media_type="text/html"
    )

@app.post("/process-text/")
async def process_text(request: Request, text_to_translate: str = Form(...), target_lang: str = Form(...)):
    # 텍스트 수정 및 번역
    corrected_text = correct_spacing(text_to_translate)
    final_text = correct_typo(corrected_text, model, tokenizer)
    translated_text2 = translate_text(final_text, dest_language=target_lang)

    # HTML 템플릿에 번역된 텍스트 전달
    return templates.TemplateResponse(
        "index.html",
        {"translated_text2": translated_text2[0], "request": request},
        media_type="text/html"
    )
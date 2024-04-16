
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from paddleocr import PaddleOCR
from pykospacing import Spacing
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = FastAPI()

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
    tokenizer.src_lang = "ko_KR"
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

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...), target_lang: str = Form(...)):
    image_data = await file.read()
    image = read_imagefile(image_data)
    cv2.imwrite("temp.jpg", image)  # 임시 파일 저장

    result = ocr.ocr("temp.jpg", cls=False)
    extracted_text = " ".join([line[1][0] for line in result[0]])
    corrected_text = correct_spacing(extracted_text)
    final_text = correct_typo(corrected_text, model, tokenizer)
    translated_text = translate_text(final_text, dest_language=target_lang)

    return JSONResponse(content={"사진에서 텍스트 추출": extracted_text,
                                 "맞춤법 조정": final_text,
                                 "번역글": translated_text})







    



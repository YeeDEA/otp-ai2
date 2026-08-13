# -*- coding: utf-8 -*-
"""Gemini (Vertex AI) post-correction of OCR chat-log output.

Extracted verbatim from notebooks/kakao_ocr_easyocr_gemini_correction.ipynb.
In the notebook the JSON key path pointed at a Colab upload
(/content/...json) — pass your own service-account key path.
"""

import os
import json


def run_gemini_flash(json_key_path, prompt_text):
    """
    JSON 키 파일을 사용하여 인증하고 Gemini 모델을 실행하는 함수
    """
    import vertexai
    from vertexai.generative_models import GenerativeModel

    # 1. JSON 파일 존재 확인
    if not os.path.exists(json_key_path):
        raise FileNotFoundError(f"키 파일을 찾을 수 없습니다: {json_key_path}")

    # 2. JSON 파일에서 프로젝트 ID 추출 (Service Account JSON 구조 가정)
    try:
        with open(json_key_path, 'r') as f:
            key_data = json.load(f)
            project_id = key_data.get("project_id")

        if not project_id:
            raise ValueError("JSON 파일에 'project_id' 정보가 없습니다.")

    except Exception as e:
        print(f"JSON 파일 읽기 오류: {e}")
        return

    # 3. 인증 환경 변수 설정 (Google Cloud 인증 표준 방식)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_key_path

    # 4. Vertex AI 초기화
    # location은 리전(Region)을 의미합니다. (예: us-central1, asia-northeast3 등)
    vertexai.init(project=project_id, location="us-central1")

    # 5. 모델 로드
    # 주의: Gemini 2.5 Flash가 출시되면 모델 이름을 해당 버전으로 변경하세요.
    # 예: "gemini-2.5-flash-preview" 또는 공식 명칭
    model_name = "gemini-2.5-flash"

    try:
        model = GenerativeModel(model_name)

        # 6. 콘텐츠 생성 (답변 받기)
        print(f"--- 입력 문장: {prompt_text} ---")
        response = model.generate_content(prompt_text)

        return response.text

    except Exception as e:
        return f"모델 실행 중 오류 발생: {e}"

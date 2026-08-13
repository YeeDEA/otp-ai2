# -*- coding: utf-8 -*-
"""OCR engine wrappers (EasyOCR / PaddleOCR / RapidOCR).

Extracted from:
- notebooks/kakao_parser_easyocr.ipynb  — load_easyocr (local-weights EasyOCR
  setup, originally module-level code) and parse_chat_image (EasyOCR ->
  PaddleOCR-style dict). In the notebook the reader was a module-level global
  `ocr`; here it is passed as a parameter.
- notebooks/paddleocr_speed_test.ipynb  — create_paddle_ocr (speed-tuned
  init, originally module-level) and paddle_parse_chat_image (resize + OCR).
- notebooks/rapidocr_test.ipynb         — run_rapidocr and fast_cpu_ocr.

Heavy engine imports are done inside each function so that importing this
module does not require every OCR backend to be installed.
"""

import os
import shutil
import time

import cv2
import numpy as np


# =========================================================
# EasyOCR (from kakao_parser_easyocr.ipynb)
# =========================================================

def safe_copy(src, dst):
    """
    소스 파일이 존재할 경우 대상 경로로 복사합니다.
    이미 대상 파일이 존재하면 복사하지 않습니다.

    Args:
        src (str): 원본 파일 경로
        dst (str): 대상 파일 경로
    """
    if not os.path.exists(dst):
        # print(f"모델 복사: {src} → {dst}")
        shutil.copy(src, dst)
    else:
        pass
        # print(f"✔ 이미 존재함: {dst}")


def load_easyocr(detector_src="weights/easyocr/craft_mlt_25k.pth",
                 recog_src="weights/easyocr/korean_g2.pth"):
    """
    EasyOCR 모델 파일을 홈 디렉토리(~/.EasyOCR)로 복사하고 Reader를 초기화합니다.
    (원본 노트북에서는 모듈 최상단에서 실행되던 코드)

    Returns:
        easyocr.Reader: 한국어/영어 Reader (gpu=False, download_enabled=False)
    """
    import easyocr

    base_dir = os.path.expanduser("~/.EasyOCR")
    detector_dst = os.path.join(base_dir, "model", "craft_mlt_25k.pth")
    recog_dst = os.path.join(base_dir, "model", "korean_g2.pth")

    # 모델 저장 경로 생성
    os.makedirs(os.path.join(base_dir, "model"), exist_ok=True)

    # 모델 파일 복사 실행
    safe_copy(detector_src, detector_dst)
    safe_copy(recog_src, recog_dst)

    print("EasyOCR 모델 로딩 중...")
    ocr = easyocr.Reader(
        ['ko', 'en'],
        gpu=False,
        download_enabled=False
    )
    print("로컬 EasyOCR 로딩 완료!")
    return ocr


def parse_chat_image(img_path, ocr):
    """
    이미지 파일을 로드하여 EasyOCR을 수행하고, 결과를 PaddleOCR 포맷과 유사하게 반환합니다.

    Args:
        img_path (str): 분석할 이미지 파일 경로
        ocr (easyocr.Reader): load_easyocr()로 만든 Reader
            (노트북에서는 전역 변수였음)

    Returns:
        dict: {
            "rec_texts": [텍스트 리스트],
            "rec_polys": [바운딩 박스 좌표 리스트],
            "width": 이미지 너비
        }
        (오류 시 에러 메시지 문자열 반환)
    """
    # 한글 경로 지원을 위해 numpy로 파일 읽기
    img_array = np.fromfile(img_path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return "Error: 이미지 없음"

    h, w = img.shape[:2]

    # EasyOCR 실행 (detail=1: 좌표 포함)
    result = ocr.readtext(img, detail=1)

    # 결과 데이터 구조 변환
    rec_texts = []
    rec_polys = []

    for box, text, conf in result:
        rec_texts.append(text)
        rec_polys.append(box)

    return {
        "rec_texts": rec_texts,
        "rec_polys": rec_polys,
        "width": w
    }


# =========================================================
# PaddleOCR (from paddleocr_speed_test.ipynb)
# =========================================================

def create_paddle_ocr():
    """
    속도 최적화 설정으로 PaddleOCR을 초기화합니다.
    (원본 노트북에서는 모듈 최상단에서 실행되던 코드 — 프로그램 시작 시 한 번만 로딩)
    """
    from paddleocr import PaddleOCR

    print("🔄 모델을 메모리에 로딩 중입니다... (최초 1회)")
    ocr = PaddleOCR(
        lang='korean',
        use_angle_cls=False,     # [속도 핵심] 문서가 회전되지 않았다면 False 추천
        enable_mkldnn=True,      # [속도 핵심] CPU 가속 활성화
    )
    print("✅ 모델 로딩 완료!")
    return ocr


def paddle_parse_chat_image(img_path, ocr):
    """
    이미지 읽기(한글 경로 대응) → 1080px 리사이징 → PaddleOCR 수행.
    from notebooks/paddleocr_speed_test.ipynb (parse_chat_image).

    Args:
        img_path (str): 이미지 경로
        ocr: create_paddle_ocr()로 만든 PaddleOCR 인스턴스 (노트북에서는 전역 변수)
    """
    start_time = time.time()

    # 3-1. 이미지 읽기 (한글 경로 대응)
    try:
        img_array = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception:
        return "Error: 이미지를 읽을 수 없습니다."

    if img is None: return "Error: 이미지가 없습니다."

    # 3-2. 리사이징 (속도 최적화: 960px)
    h, w, _ = img.shape
    max_side = 1080
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    current_width = img.shape[1]
    center_x = current_width / 2

    # 3-3. OCR 수행
    result = ocr.ocr(img)
    print(f"🚀 OCR 처리 소요시간: {time.time() - start_time:.4f}초")
    return result

    # NOTE: 노트북 원본에서도 return 뒤라 도달하지 않음 (원본 유지)
    if not result or not result[0]:
        return "텍스트를 찾을 수 없습니다."


def fast_cpu_ocr(img_path):
    """CPU 가속 PaddleOCR 단발 실행. from notebooks/rapidocr_test.ipynb."""
    from paddleocr import PaddleOCR

    # ------------------------------------------------------------------
    # 1. 모델 초기화 (CPU 가속 설정)
    # ------------------------------------------------------------------
    ocr = PaddleOCR(
        lang='korean',
        use_gpu=False,           # GPU 끔
        enable_mkldnn=True,      # [핵심] CPU 연산 가속 활성화
        use_angle_cls=False,     # [핵심] 각도 분류 끔 (문서가 정방향이면 끄는 게 훨씬 빠름)
        ocr_version='PP-OCRv4',  # 최신 경량 모델 사용
        show_log=False           # 불필요한 로그 출력 방지
    )

    # ------------------------------------------------------------------
    # 2. 이미지 전처리 (리사이징)
    # ------------------------------------------------------------------
    start_time = time.time()

    img = cv2.imread(img_path)
    if img is None:
        print("이미지를 읽을 수 없습니다.")
        return

    # 원본 이미지 크기 확인
    h, w, _ = img.shape

    # [핵심] 이미지의 긴 변을 제한 (예: 960px ~ 1280px)
    # 너무 크면 CPU가 힘들어하고, 너무 작으면 인식이 안 됩니다.
    # 속도가 최우선이라면 960, 정확도와 타협하면 1280 추천
    max_side_limit = 960

    if max(h, w) > max_side_limit:
        scale = max_side_limit / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------
    # 3. 인식 수행
    # ------------------------------------------------------------------
    # cls=False를 줘서 추론 단계에서도 방향 탐지를 확실히 배제
    result = ocr.ocr(img, cls=False)

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"🚀 처리 완료: {elapsed:.4f}초 소요 (해상도: {img.shape[1]}x{img.shape[0]})")

    # 결과 출력
    if result and result[0]:
        for line in result[0]:
            print(f"텍스트: {line[1][0]}")
    else:
        print("검출된 텍스트가 없습니다.")


# =========================================================
# RapidOCR (from rapidocr_test.ipynb)
# =========================================================

def run_rapidocr(image_path,
                 det_model_path='ch_PP-OCRv4_det_infer.onnx',
                 rec_model_path='model.onnx',
                 rec_keys_path='korean_dict.txt'):
    """
    RapidOCR(onnxruntime)로 이미지 OCR 실행 후 결과를 출력합니다.
    from notebooks/rapidocr_test.ipynb (run_ocr).

    주의: 모델 파일들이 실제로 존재해야 합니다.
    """
    from rapidocr_onnxruntime import RapidOCR

    # 1. RapidOCR 엔진 초기화 (사용자 지정 모델 경로 설정)
    engine = RapidOCR(
        det_model_path=det_model_path,    # 텍스트 위치 감지 모델
        rec_model_path=rec_model_path,    # [중요] 한국어 텍스트 인식 모델
        rec_keys_path=rec_keys_path       # [중요] 한국어 사전 파일
    )

    print(f"[-] 이미지 처리 중: {image_path}")

    # 2. 이미지 읽기 및 추론 실행
    # result: 감지된 텍스트 정보 리스트, elapse: 소요 시간
    result, elapse = engine(image_path)

    if not result:
        print("[!] 텍스트를 찾을 수 없습니다.")
        return

    # 3. 결과 출력
    print(f"[-] 처리 시간: {elapse}초")
    print("-" * 30)

    # result 포맷: [[박스 좌표], '텍스트', 신뢰도]
    for i, item in enumerate(result):
        box, text, confidence = item
        print(f"[{i+1}] 텍스트: {text} (신뢰도: {confidence:.4f})")
        # print(f"    좌표: {box}") # 좌표가 필요하면 주석 해제

    print("-" * 30)

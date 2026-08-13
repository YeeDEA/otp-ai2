# -*- coding: utf-8 -*-
"""Instagram DM screenshot parser on top of PaddleOCR.

Extracted verbatim from notebooks/insta_dm_parser_paddle.ipynb.
InstagramParser runs OCR; InstagramParser2 turns the OCR dict into a
speaker-attributed log using the left/right position of each text box.
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR


class InstagramParser:
    def __init__(self):
        # PaddleOCR 초기화 (한 번만 로드)
        self.ocr = PaddleOCR(lang='korean', use_angle_cls=False)

    def run(self, image_path):
        """
        인스타그램 DM 파싱
        - 화면 중앙 기준: 왼쪽(상대방), 오른쪽(나)
        - 높이 기반 이름 감지: 왼쪽 텍스트 중 유독 높이가 작은 항목을 이름으로 식별
        """
        # 1. 이미지 읽기
        try:
            img_array = np.fromfile(image_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            return f"Error: 이미지를 읽을 수 없습니다. ({e})"

        if img is None:
            return "Error: 이미지가 유효하지 않습니다."

        height, width, _ = img.shape
        center_x = width / 2

        # 2. OCR 수행
        result = self.ocr.ocr(img)
        return result
        # NOTE: 아래 코드는 노트북 원본에서도 return 뒤라 도달하지 않음 (원본 유지)
        if not result or not result[0]:
            return "텍스트를 찾을 수 없습니다."


class InstagramParser2:
    def __init__(self):
        pass

    def run(self, ocr_result, image_width=None):
        # 1. 데이터 검증
        if 'rec_texts' not in ocr_result or 'rec_polys' not in ocr_result:
            return "Error: 데이터 형식이 올바르지 않습니다. 'rec_texts'와 'rec_polys' 키가 필요합니다."

        texts = ocr_result['rec_texts']
        polys = ocr_result['rec_polys']

        if len(texts) != len(polys):
            return "Error: 텍스트와 좌표의 개수가 일치하지 않습니다."

        # 2. 이미지 너비 자동 추정 (화자 구분의 기준점 설정을 위해)
        if image_width is None:
            max_x = 0
            for poly in polys:
                # poly가 numpy array인지 리스트인지 확인하여 처리
                np_poly = np.array(poly) if not isinstance(poly, np.ndarray) else poly
                current_max = np.max(np_poly[:, 0])
                if current_max > max_x:
                    max_x = current_max
            # 가장 오른쪽 좌표에 약간의 여유를 두어 전체 너비 추정
            image_width = max_x * 1.1

        center_x = image_width / 2

        # 3. 데이터 가공 (중심 좌표 계산)
        parsed_data = []
        for text, poly in zip(texts, polys):
            # 텍스트가 없으면 건너뜀
            if not text.strip():
                continue

            np_poly = np.array(poly) if not isinstance(poly, np.ndarray) else poly

            # X축 중심 (좌/우 화자 구분용)
            x_min = np.min(np_poly[:, 0])
            x_max = np.max(np_poly[:, 0])
            cx = (x_min + x_max) / 2

            # Y축 중심 (대화 순서 정렬용)
            y_min = np.min(np_poly[:, 1])
            y_max = np.max(np_poly[:, 1])
            cy = (y_min + y_max) / 2

            parsed_data.append({
                'text': text,
                'cx': cx,
                'cy': cy
            })

        # 4. Y축 기준 정렬 (시간 순서대로 나열)
        parsed_data.sort(key=lambda x: x['cy'])

        # 5. 화자 분류 및 로그 생성
        chat_logs = []
        for item in parsed_data:
            # 화면 중앙보다 오른쪽에 있으면 '나'
            if item['cx'] > center_x:
                speaker = "나"
            # 화면 중앙보다 왼쪽에 있으면 '상대방'
            else:
                speaker = "상대방"

            chat_logs.append(f"{speaker} : {item['text']}")

        return "\n".join(chat_logs)

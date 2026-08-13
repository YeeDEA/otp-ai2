# oTP — 스크린샷/영상 → 대화 텍스트 OCR 파이프라인 (GDGoC Yonsei)

카카오톡·인스타그램 DM 스크린샷과 스크롤 녹화 영상을 OCR로 읽어 대화 로그(발화자·시간 포함)로 복원하는 파이프라인.
GDGoC Yonsei oTP 프로젝트 (2025-12 인기상 수상). OCR 엔진 비교(PaddleOCR/EasyOCR/RapidOCR/Pororo) 끝에 PaddleOCR 채택, SPyNet(광학 흐름)으로 스크롤 영상을 스크린샷으로 분해.
관련 레포: `2022148084/otp-ai`, `YeeDEA/otp-ai2`

**작업 기간: 2025-11-16 ~ 2025-12-03** (파일 작성 시각 기준)

| 파일 | 원본 파일명 | 작성일 | 내용 |
|---|---|---|---|
| `kakao_ocr_easyocr_early_experiment.ipynb` | B/Untitled6.ipynb | 2025-11-16 | 카톡 스크린샷 EasyOCR 파싱 초기 실험 |
| `kakao_ocr_paddle_chatlog.ipynb` | A/Untitled15.ipynb | 2025-11-17 | 카톡 스크린샷 PaddleOCR 채팅로그 변환 |
| `kakao_ocr_easyocr_gemini_correction.ipynb` | A/Untitled16.ipynb | 2025-11-25 | EasyOCR + Gemini LLM 오류 보정 |
| `rapidocr_test.ipynb` | (동일) | 2025-11-27 | RapidOCR 한국어 인식 테스트 |
| `kakao_parser_easyocr.ipynb` | A/Untitled22.ipynb | 2025-11-30 | EasyOCR 기반 카톡 대화 파서 모듈 |
| `spynet_scroll_video_pipeline.ipynb` | A/Untitled23.ipynb | 2025-11-30 | SPyNet 스크롤 영상→스크린샷→OCR 전체 파이프라인 |
| `insta_dm_parser_paddle.ipynb` | A/Untitled24.ipynb | 2025-11-30 | 인스타그램 DM 파서 (PaddleOCR) |
| `img_to_text.ipynb` | (동일) | 2025-11-30 | 메인 진입점: 카카오/인스타 분기 판별 후 OCR+파싱 실행 (초안) |
| `paddleocr_speed_test.ipynb` | (동일) | 2025-12-02 | PaddleOCR 속도 최적화(mkldnn·리사이징) + 좌표 기반 카톡 파서 |
| `ocr_engine_comparison.ipynb` | A/Untitled26.ipynb | 2025-12-03 | OCR 3종 비교 차트 — PaddleOCR 선정 근거 |

## 비고
- 스크린샷 출처 분류기(MobileNetV2)는 별도 폴더 `kakao-insta-detector`로 분리 (YeeDEA/kakao_insta_detector 레포 대응)
- 제외: A/Untitled19.ipynb (Pororo OCR 설치 실패 스크래치)

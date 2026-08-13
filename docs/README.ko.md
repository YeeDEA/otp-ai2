# oTP — 스크린샷/영상 → 대화 텍스트 OCR 파이프라인 (GDGoC Yonsei)

카카오톡·인스타그램 DM 스크린샷과 스크롤 녹화 영상을 OCR로 읽어 대화 로그(발화자·시간 포함)로 복원하는 파이프라인.
GDGoC Yonsei oTP 프로젝트 (2025-12 인기상 수상). OCR 엔진 검토(PaddleOCR·EasyOCR·RapidOCR 3종 비교 + Pororo는 설치 실패로 탈락) 끝에 PaddleOCR 채택, SPyNet(광학 흐름)으로 스크롤 영상을 스크린샷으로 분해.
관련 레포: `2022148084/otp-ai`, `YeeDEA/otp-ai2`

**작업 기간: 2025-11-16 ~ 2025-12-03** (파일 작성 시각 기준)

| 파일 | 원본 파일명 | 작성일 | 내용 |
|---|---|---|---|
| `kakao_ocr_easyocr_early_experiment.ipynb` | B/Untitled6.ipynb | 2025-11-16 | 카톡 스크린샷 EasyOCR 파싱 초기 실험 |
| `kakao_ocr_paddle_chatlog.ipynb` | A/Untitled15.ipynb | 2025-11-17 | 카톡 스크린샷 PaddleOCR 채팅로그 변환 |
| `kakao_ocr_easyocr_gemini_correction.ipynb` | A/Untitled16.ipynb | 2025-11-25 | EasyOCR + Gemini LLM 오류 보정 |
| `pororo_install_failed.ipynb` | A/Untitled19.ipynb | 2025-11-25 | Pororo OCR 설치 시도 → 실패(`torch==1.6.0` 핀 충돌, `ResolutionImpossible`). Pororo를 시도했다는 유일한 기록 |
| `rapidocr_test.ipynb` | (동일) | 2025-11-27 | RapidOCR 한국어 인식 테스트 |
| `kakao_parser_easyocr.ipynb` | A/Untitled22.ipynb | 2025-11-30 | EasyOCR 기반 카톡 대화 파서 모듈 |
| `spynet_scroll_video_pipeline.ipynb` | A/Untitled23.ipynb | 2025-11-30 | SPyNet 스크롤 영상→스크린샷→OCR 전체 파이프라인 |
| `insta_dm_parser_paddle.ipynb` | A/Untitled24.ipynb | 2025-11-30 | 인스타그램 DM 파서 (PaddleOCR) |
| `img_to_text.ipynb` | (동일) | 2025-11-30 | 메인 진입점: 카카오/인스타 분기 판별 후 OCR+파싱 실행 (초안) |
| `paddleocr_speed_test.ipynb` | (동일) | 2025-12-02 | PaddleOCR 속도 최적화(mkldnn·리사이징) + 좌표 기반 카톡 파서 |
| `ocr_engine_comparison.ipynb` | A/Untitled26.ipynb | 2025-12-03 | OCR 3종 비교 차트 — PaddleOCR 선정 근거 |

## 비고
- 스크린샷 출처 분류기(MobileNetV2)는 별도 폴더 `kakao-insta-detector`로 분리 (YeeDEA/kakao_insta_detector 레포 대응)
- A/Untitled19.ipynb는 처음에 "스크래치"로 제외했으나, **Pororo 시도의 유일한 증거**이므로
  `notebooks/pororo_install_failed.ipynb`로 편입함(위 표 참조).
- Pororo는 정확도·속도로 탈락한 것이 아니라 **설치 단계에서 탈락**함. 비교 차트의 3종(EasyOCR·RapidOCR·PaddleOCR)과
  탈락 사유의 종류가 다르며, Pororo에 대한 정확도·속도 수치는 이 프로젝트 어디에도 없음.

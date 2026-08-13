"""otp_ocr — chat-screenshot OCR parsing modules extracted from the Colab notebooks.

The notebooks under notebooks/ are the original experiments; this package is the
same code lifted into importable modules (no functional changes intended).

Modules:
- kakao: coordinate-based KakaoTalk screenshot parser (kakao_parser_easyocr.ipynb)
- insta: Instagram DM parser (insta_dm_parser_paddle.ipynb)
- engines: OCR engine wrappers — EasyOCR / PaddleOCR / RapidOCR
- scroll_video: SPyNet optical-flow scroll-video frame extraction
- gemini_correction: Vertex AI Gemini post-correction of OCR output
"""

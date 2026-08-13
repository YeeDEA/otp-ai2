# -*- coding: utf-8 -*-
"""CLI: KakaoTalk screenshot -> chat log.

Mirrors the __main__ flow of notebooks/kakao_parser_easyocr.ipynb:
EasyOCR (local weights) -> coordinate-based parser.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from otp_ocr.engines import load_easyocr, parse_chat_image
from otp_ocr.kakao import parse_kakao_dict


def main():
    ap = argparse.ArgumentParser(description="KakaoTalk screenshot -> chat log (EasyOCR)")
    ap.add_argument("image", help="path to the KakaoTalk screenshot")
    ap.add_argument("--detector", default="weights/easyocr/craft_mlt_25k.pth",
                    help="EasyOCR CRAFT detector weights")
    ap.add_argument("--recognizer", default="weights/easyocr/korean_g2.pth",
                    help="EasyOCR korean_g2 recognizer weights")
    args = ap.parse_args()

    ocr = load_easyocr(args.detector, args.recognizer)

    print("OCR + 파싱 시작:", args.image)
    ocr_pack = parse_chat_image(args.image, ocr)
    if isinstance(ocr_pack, str):
        print(ocr_pack)
        return
    parsed_log = parse_kakao_dict(ocr_pack, ocr_pack["width"])

    print("\n====================================")
    print(" 최종 변환 결과")
    print("====================================")
    print(parsed_log)


if __name__ == "__main__":
    main()

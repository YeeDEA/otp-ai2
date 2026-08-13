# -*- coding: utf-8 -*-
"""CLI: Instagram DM screenshot -> chat log.

Mirrors the __main__ flow of notebooks/insta_dm_parser_paddle.ipynb:
PaddleOCR -> left/right position-based speaker attribution.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from otp_ocr.insta import InstagramParser, InstagramParser2


def main():
    ap = argparse.ArgumentParser(description="Instagram DM screenshot -> chat log (PaddleOCR)")
    ap.add_argument("image", help="path to the Instagram DM screenshot")
    args = ap.parse_args()

    parser = InstagramParser()
    print(f"Analyzing {args.image}...")
    result_text = parser.run(args.image)
    if isinstance(result_text, str):
        print(result_text)
        return

    parser2 = InstagramParser2()
    parsed_log = parser2.run(result_text[0])

    print("-" * 40)
    print("인스타그램 대화 파싱 결과")
    print("-" * 40)
    print(parsed_log)


if __name__ == "__main__":
    main()

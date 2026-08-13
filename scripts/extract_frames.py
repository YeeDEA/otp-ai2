# -*- coding: utf-8 -*-
"""CLI: scroll-recording video -> screenshots via SPyNet optical flow.

Mirrors the extract_screenshots step of
notebooks/spynet_scroll_video_pipeline.ipynb. Requires the SPyNet weights
at weights/spynet/network-sintel-final.pytorch (run from the repo root).
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from otp_ocr.scroll_video import extract_screenshots


def main():
    ap = argparse.ArgumentParser(description="Scroll video -> screenshots (SPyNet)")
    ap.add_argument("video", help="path to the scroll-recording video")
    ap.add_argument("--out", default=None,
                    help="screenshot output dir (default: ./runs/<timestamp>/screenshots)")
    args = ap.parse_args()

    out_dir = args.out
    if out_dir is None:
        folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("./runs", folder_name, "screenshots")

    print(f"Extracting screenshots to: {out_dir}")
    extract_screenshots(args.video, out_dir)


if __name__ == "__main__":
    main()

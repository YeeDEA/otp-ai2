# OCR strategy

> Source: the team's strategy presentation for the oTP project (GDGoC Yonsei), "KakaoTalk OCR strategy deck".

This is a faithful English summary of the design intent recorded in the deck, annotated with where each
decision actually lands in this repo. The deck is a short draft (four slides); where it stops short, that is
noted rather than filled in. Measured numbers do not appear in the deck at all — the numbers in this project
come from the notebooks, and are kept in the [README](../README.md).

## Problem framing

The deck states the goal as converting **unstructured image data into a structured text asset**: a scroll-captured
KakaoTalk conversation should be automatically turned into analyzable, structured data. Its framing line is that
an image cannot be read (by downstream analysis), but data can be analyzed — so the value of the system is the
conversion step itself, not the OCR output as such.

The input is described as a KakaoTalk capture (a screenshot of a conversation, scroll-captured); the output as
structured text data, time-ordered.

## Pipeline as stated in the deck

The deck presents the system as five stages:

1. **Input** — conversation capture image.
2. **Preprocessing** — UI removal and region-of-interest (ROI) segmentation.
3. **Filtering** — exception handling for unneeded information such as system messages.
4. **Analysis** — speaker identification and handling of consecutive utterances from the same speaker,
   described explicitly as *state management*.
5. **Output** — chronologically ordered conversation data.

A dedicated slide expands stage 2: the ROI is set so that the **static header** of the chat screen — chat-room
information and the notice/announcement banner — is excluded from the recognized region. The slide's illustration
is a placeholder for a full screenshot with a callout marking that header band; no actual conversation image is
embedded in the deck file.

## Architectural decisions in the deck

- **Structure comes from layout, not from language.** Speaker identification is treated as a positional problem
  handled after OCR, not something the OCR engine provides. The pipeline therefore separates recognition from
  parsing.
- **Noise is removed in two different places for two different reasons.** Geometric noise (the static header/notice
  band) is cut in preprocessing by restricting the ROI; semantic noise (system messages such as join/leave notices)
  is cut later by content filtering. The deck keeps these as distinct stages.
- **Turn reconstruction is stateful.** Consecutive utterances by the same speaker are not independent records;
  the parser carries state across recognized lines to group them into one turn.
- **Chronological ordering is the output contract.** The deliverable is a time-sorted conversation log, which is
  what makes the result analyzable.

## Engine selection

The deck does **not** discuss OCR engines, and contains no engine comparison, no accuracy figures, and no timing
figures. The engine decision (EasyOCR / RapidOCR / PaddleOCR / Pororo, with PaddleOCR selected) is documented
only in the notebooks, and is summarized in the README's OCR engine comparison section. Nothing in this document
should be read as deck-sourced evidence for that choice.

## How the deck maps onto this repo

| Deck stage | Where it lives |
|---|---|
| Preprocessing — ROI, static header exclusion | `src/otp_ocr/kakao.py`: a vertical start threshold derived from image height, raised further by a first pass that detects a left-anchored notice/banner box near the top |
| Filtering — system messages, UI elements | `src/otp_ocr/kakao.py`: keyword, regex, and UI-label rules that drop join/leave/invite notices, deleted-message markers, gift/transfer notices, and button labels; also quoted-reply blocks |
| Analysis — speaker identification | `src/otp_ocr/kakao.py`, `src/otp_ocr/insta.py`: bubble position relative to the image center X assigns the turn to self vs. the other party; on the left side, a short leading line is treated as a sender name |
| Analysis — consecutive utterances (state) | `src/otp_ocr/kakao.py`: recognized text lines accumulate into an open turn, which is closed and emitted when a timestamp line is reached |
| Output — chronological data | `src/otp_ocr/kakao.py`: boxes sorted by vertical center; date headers update a running date, timestamps normalized to 24-hour form |

## Where the repo goes beyond the deck

The deck covers the screenshot-to-log path for KakaoTalk only. The repo additionally implements:

- an Instagram DM parser on the same left/right positional principle,
- scroll-recording **video** input decomposed into screenshots via SPyNet optical flow,
- an LLM post-correction pass over OCR output,
- the OCR engine comparison that motivated the engine choice.

These are notebook-derived, not deck-derived.

## Privacy note

The deck contains image *placeholders* rather than embedded conversation screenshots, and no personal data was
carried from it into this repo. Any sample chat screenshots used during development are not committed here.

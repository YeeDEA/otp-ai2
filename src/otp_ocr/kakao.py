# -*- coding: utf-8 -*-
"""KakaoTalk screenshot -> chat-log parser (coordinate-based).

Extracted verbatim from notebooks/kakao_parser_easyocr.ipynb (the most refined
version of the parser). An earlier, slightly different variant of
parse_kakao_dict lives in notebooks/paddleocr_speed_test.ipynb — differences:
notice-banner x_left threshold (50 vs 100 px), the scroll-date filter there does
not check x position, and the "에게 답장" removal only looks one item ahead.

Input format: a dict with 'rec_texts' (list of strings) and 'rec_polys'
(list of 4-point boxes), i.e. PaddleOCR-style output; engines.easyocr_to_pack
produces the same shape from EasyOCR.
"""

import re
import numpy as np
from datetime import datetime, timedelta


# =========================================================
# 1. 유틸리티 함수 (전처리 및 필터링)
# =========================================================

def clean_text(text):
    """
    OCR 인식 결과에서 불필요한 특수문자(파이프, 역슬래시 등)를 제거합니다.

    Args:
        text (str): 원본 텍스트

    Returns:
        str: 정제된 텍스트
    """
    text = re.sub(r"[|│]", "", text)
    return text.strip()


def is_noise(text):
    """
    주어진 텍스트가 대화 내용이 아닌 시스템 메시지나 UI 요소인지 판별합니다.

    Args:
        text (str): 판별할 텍스트

    Returns:
        bool: 노이즈이면 True, 유효한 대화이면 False
    """
    if not text:
        return True

    # 1. 절대 제외 키워드 (포함 시 무조건 노이즈)
    exclude_keywords = ["공유하기", "들어왔습니다", "나갔습니다", "초대했습니다"]
    for keyword in exclude_keywords:
        if keyword in text:
            return True

    # 2. 시스템 메시지 패턴 (정규식)
    system_patterns = [
        r".*님이 들어왔습니다\.$",
        r".*님이 나갔습니다\.$",
        r".*님이 .*님을 초대했습니다\.$",
        r"^삭제된 메시지입니다\.$",
        r".*기프티콘을 보냈습니다.*",
        r".*송금했습니다.*",
        r"^톡게시판.*",
    ]
    for p in system_patterns:
        if re.match(p, text):
            return True

    # 3. UI 버튼 및 키워드
    ui_keywords = {
        "사진","동영상","음성메시지","보이스톡","페이스톡","라이브톡",
        "선물하기","송금","정산하기","프로필 보기","공지 등록","좋아요","공감",
        "안읽음","MY","채팅방","메뉴","전송","읽음"
    }
    if text in ui_keywords:
        return True

    # 4. 숫자만 있는 경우 (읽지 않음 표시 등)
    if text.replace(':','').isdigit() and len(text) < 3:
        return True

    return False


def format_time(ts_str):
    """
    OCR로 인식된 시간 문자열을 표준 24시간 포맷(HH:MM)으로 변환합니다.
    (예: '오후9:24' -> '21:24')

    Args:
        ts_str (str): 원본 시간 문자열

    Returns:
        str: 'HH:MM' 형식의 문자열 (변환 실패 시 원본 반환)
    """
    ts_str = ts_str.replace(" ", "")
    final_time = ts_str

    try:
        is_pm = "오후" in ts_str
        is_am = "오전" in ts_str

        # 시간 패턴 추출 (숫자:숫자 또는 숫자.숫자)
        m = re.search(r"(\d{1,2})[:\.,]?(\d{2})", ts_str)
        if m:
            hour, minute = int(m.group(1)), int(m.group(2))

            # 오전/오후 보정
            if is_pm and hour != 12:
                hour += 12
            if is_am and hour == 12:
                hour = 0
            final_time = f"{hour:02}:{minute:02}"
    except:
        pass

    return final_time


# =========================================================
# 2. 카카오톡 대화 파싱 로직 (핵심)
# =========================================================

def parse_kakao_dict(ocr_result, image_width=720, image_height=None):
    """
    OCR 결과를 분석하여 카카오톡 대화 로그 텍스트로 변환합니다.
    위치(좌표) 정보를 기반으로 줄바꿈, 발화자(나/상대방), 타임스탬프 등을 추론합니다.

    Args:
        ocr_result (dict): 'rec_texts', 'rec_polys'를 포함한 OCR 결과 딕셔너리
        image_width (int): 이미지의 너비
        image_height (int, optional): 이미지의 높이 (없을 경우 좌표로 추정)

    Returns:
        str: 변환된 대화 로그 문자열
    """
    if 'rec_texts' not in ocr_result or 'rec_polys' not in ocr_result:
        return "Error: 데이터 형식이 올바르지 않습니다."

    texts = ocr_result['rec_texts']
    polys = ocr_result['rec_polys']
    center_x = image_width / 2

    # 이미지 높이 추정 (입력이 없을 경우 폴리곤 좌표 기반)
    if image_height is None:
        all_ys = []
        for p in polys:
            np_p = np.array(p)
            all_ys.extend(np_p[:, 1])
        image_height = max(all_ys) if all_ys else 2000

    # -------------------------------------------------------------
    # [설정] 영역 필터링 기준값
    # -------------------------------------------------------------
    y_start_threshold = image_height / 8    # 상단 헤더 영역 무시 기준
    notice_search_limit = image_height / 4  # 공지사항 검색 범위

    # 정규식 정의 (시간 및 날짜 패턴)
    time_regexs = [
        re.compile(r".*[\d]{1,2}:[\d]{2}$"),
        re.compile(r".*[\d]{1,2}\.[\d]{2}$"),
        re.compile(r".*[\d]{1,2},[\d]{2}$"),
        re.compile(r".*오[전후]\s*\d{3,4}$")
    ]
    date_regex = re.compile(r"^20\d{2}[.년]\s*\d{1,2}[.월]\s*\d{1,2}[.일]?.*")

    # 스크롤 시 나타나는 날짜 패턴
    scroll_date_regex = re.compile(r"^\d{1,2}[\.\s]+\d{1,2}[\.\s]+[월화수목금토일]")

    # -------------------------------------------------------------
    # 1차 패스: 공지사항 위치 감지 및 y_start_threshold 조정
    # -------------------------------------------------------------
    for text, poly in zip(texts, polys):
        try:
            np_poly = np.array(poly)
            y_max = np.max(np_poly[:, 1])
            y_center = (np.min(np_poly[:, 1]) + y_max) / 2
            x_left = np.min(np_poly[:, 0])
        except: continue

        # 화면 왼쪽에 바짝 붙어있는 텍스트는 공지사항/배너로 간주
        if y_center < notice_search_limit and x_left < 100:
            if y_max > y_start_threshold:
                y_start_threshold = y_max + 5

    raw_items = []

    # -------------------------------------------------------------
    # 2차 패스: 데이터 필터링 및 텍스트 아이템 생성
    # -------------------------------------------------------------
    for text, poly in zip(texts, polys):
        text = clean_text(text)
        if not text or is_noise(text): continue

        try:
            np_poly = np.array(poly)
            y_center = (np.min(np_poly[:, 1]) + np.max(np_poly[:, 1])) / 2
            x_left = np.min(np_poly[:, 0])
            x_right = np.max(np_poly[:, 0])
            x_center = (x_left + x_right) / 2
        except: continue

        # 상단 영역 필터링
        if y_center < y_start_threshold: continue
        # 우측 스크롤 날짜 필터링
        if scroll_date_regex.search(text) and x_center > center_x: continue

        is_timestamp = any(r.match(text) for r in time_regexs)
        is_date = bool(date_regex.match(text))

        raw_items.append({
            'text': text, 'y_center': y_center, 'x_left': x_left, 'x_right': x_right,
            'is_timestamp': is_timestamp, 'is_date': is_date
        })

    if not raw_items: return "텍스트가 없습니다."

    # Y축 기준 정렬 (필수)
    raw_items.sort(key=lambda x: x['y_center'])

    # -------------------------------------------------------------
    # "에게 답장" 패턴 및 관련 인용구 제거 로직
    # -------------------------------------------------------------
    indices_to_remove = set()
    for i in range(len(raw_items)):
        text = raw_items[i]['text']
        if "에게 답장" in text or "에게답장" in text or "에계 답장" in text:
            # 1. 답장 문구 자체 삭제
            indices_to_remove.add(i)

            current_y = raw_items[i]['y_center']

            # 2. 같은 줄에 있는 텍스트들 삭제 (이름 등)

            # 뒤쪽 탐색
            for j in range(i + 1, len(raw_items)):
                if abs(raw_items[j]['y_center'] - current_y) < 20:
                    indices_to_remove.add(j)
                else:
                    break

            # 앞쪽 탐색
            for j in range(i - 1, -1, -1):
                if abs(raw_items[j]['y_center'] - current_y) < 20:
                    indices_to_remove.add(j)
                else:
                    break

            # 3. 바로 아래 줄(인용된 메시지) 삭제
            for k in range(i + 1, len(raw_items)):
                next_item = raw_items[k]
                y_diff = next_item['y_center'] - current_y

                if y_diff < 20: # 같은 줄은 이미 처리함
                    continue

                # 일정 범위 내에 있는 다음 줄을 인용구로 간주
                if y_diff < 100:
                    if not next_item['is_timestamp'] and not next_item['is_date']:
                        indices_to_remove.add(k)
                    break
                else:
                    break

    # 필터링된 리스트 생성
    processed_items = [item for i, item in enumerate(raw_items) if i not in indices_to_remove]

    # -------------------------------------------------------------
    # 줄 그룹화 (Line Grouping)
    # -------------------------------------------------------------
    Y_TOLERANCE = 50
    grouped_lines = []

    if processed_items:
        curr_items = [processed_items[0]]
        curr_y = processed_items[0]['y_center']
        for item in processed_items[1:]:
            if abs(item['y_center'] - curr_y) < Y_TOLERANCE:
                curr_items.append(item)
            else:
                _process_and_save_groups(grouped_lines, curr_items)
                curr_items = [item]
                curr_y = item['y_center']
        _process_and_save_groups(grouped_lines, curr_items)

    # -------------------------------------------------------------
    # 날짜 정보 초기화
    # -------------------------------------------------------------
    detected_dates = []
    for line in grouped_lines:
        if line['type'] == 'date':
            text = " ".join([i['text'] for i in line['items']])
            match = re.search(r"(\d{4})[.년]\s*(\d{1,2})[.월]\s*(\d{1,2})", text)
            if match:
                y, m, d = map(int, match.groups())
                detected_dates.append(datetime(y, m, d))

    if detected_dates:
        earliest_date = min(detected_dates)
        start_date = earliest_date - timedelta(days=1)
        current_date = f"{start_date.year}. {start_date.month}. {start_date.day}."
    else:
        current_date = "2000. 1. 1."

    # -------------------------------------------------------------
    # 메인 변환 루프: 그룹 -> 텍스트 로그
    # -------------------------------------------------------------
    final_chat = []
    current_turn_lines = []

    for line in grouped_lines:
        if line['type'] == 'date':
            raw_date = " ".join([i['text'] for i in line['items']])
            match = re.search(r"(\d{4})[.년]\s*(\d{1,2})[.월]\s*(\d{1,2})", raw_date)
            if match:
                current_date = f"{match.group(1)}. {match.group(2)}. {match.group(3)}."

        elif line['type'] == 'text':
            current_turn_lines.append(line)

        elif line['type'] == 'timestamp':
            if not current_turn_lines: continue

            raw_time = " ".join([i['text'] for i in line['items']])
            time_str = format_time(raw_time)
            full_ts = f"{current_date} {time_str}"

            # 발화자 및 위치 추론
            first_line = current_turn_lines[0]
            first_min_x = min(item['x_left'] for item in first_line['items'])
            first_max_x = max(item['x_right'] for item in first_line['items'])
            first_center_x = (first_min_x + first_max_x) / 2

            speaker = ""
            messages = []

            if first_center_x > center_x:
                # [오른쪽] 나
                speaker = "나"
                for l in current_turn_lines:
                    messages.append(" ".join([i['text'] for i in l['items']]))
            else:
                # [왼쪽] 상대방
                first_text = " ".join([i['text'] for i in first_line['items']])
                # 이름일 확률 판단 (짧은 길이 + 후속 메시지 존재)
                is_likely_name = len(first_text) < 15 and len(current_turn_lines) > 1

                if is_likely_name:
                    speaker = first_text
                    start_idx = 1
                else:
                    speaker = "상대방"
                    start_idx = 0

                for l in current_turn_lines[start_idx:]:
                    messages.append(" ".join([i['text'] for i in l['items']]))

            full_message = " ".join(messages)

            if full_message.strip():
                final_chat.append(f"{full_ts}, {speaker} : {full_message}")
            elif speaker != "나" and not full_message.strip():
                 final_chat.append(f"{full_ts}, {speaker} : (사진/이모티콘)")

            current_turn_lines = []

    return "\n".join(final_chat)


def _process_and_save_groups(grouped_lines, items):
    """
    임시 그룹을 세부 분석하여 줄(Line) 단위로 분리하고 유형별로 저장합니다.

    Args:
        grouped_lines (list): 결과가 저장될 리스트
        items (list): 그룹화할 텍스트 아이템 리스트
    """
    # 1. Y 중심으로 정렬
    items.sort(key=lambda x: x['y_center'])

    # 2. 미세 라인 분리 (Sub-grouping)
    sub_lines = []
    if items:
        curr_sub = [items[0]]
        for item in items[1:]:
            # 12px 이상 차이나면 다른 줄로 간주
            if (item['y_center'] - curr_sub[-1]['y_center']) > 12:
                sub_lines.append(curr_sub)
                curr_sub = [item]
            else:
                curr_sub.append(item)
        sub_lines.append(curr_sub)

    # 3. 각 줄별로 X 정렬 후 저장
    for line_items in sub_lines:
        line_items.sort(key=lambda x: x['x_left'])
        texts = [i for i in line_items if not i['is_timestamp'] and not i['is_date']]
        times = [i for i in line_items if i['is_timestamp']]
        dates = [i for i in line_items if i['is_date']]

        if texts: grouped_lines.append({'type': 'text', 'items': texts})
        if times: grouped_lines.append({'type': 'timestamp', 'items': times})
        if dates: grouped_lines.append({'type': 'date', 'items': dates})

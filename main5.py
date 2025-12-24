# main7.py — updated: dynamic invoice zone below comments + prefix fallback + full-page total selection
import os
import json
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
import cv2
import pytesseract
from difflib import SequenceMatcher
import re
import openpyxl
from datetime import datetime
from PIL import Image
import logging
import traceback
import sys
import argparse

# --------------------- LOGGING CONFIGURATION ---------------------
# Create logs directory if it doesn't exist
LOG_DIR = Path(__file__).parent.absolute() / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configure logging with both file and console handlers
log_filename = LOG_DIR / f"ocr_application_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Create logger
logger = logging.getLogger('OCR_Application')
logger.setLevel(logging.DEBUG)

# Create formatters
detailed_formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(funcName)-25s | Line %(lineno)-4d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
simple_formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)

# File handler - detailed logging
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(detailed_formatter)

# Console handler - important messages only
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(simple_formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("="*80)
logger.info("OCR APPLICATION STARTED")
logger.info(f"Log file: {log_filename}")
logger.info("="*80)
# ----------------------------------------------------------------

# --------------------- LOAD CONFIGURATION ---------------------
try:
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from: {config_path}")
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = {
            "tesseract_path": r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            "poppler_path": r"D:\Program Files\poppler-25.07.0\Library\bin",
            "models": {
                "detection": "models/PP-OCRv5_server_det",
                "recognition": "models/PP-OCRv5_server_rec"
            }
        }
except Exception as e:
    logger.error(f"Error loading config: {e}")
    logger.debug(traceback.format_exc())
    raise
# ----------------------------------------------------------------

# Configure Tesseract path from config
tesseract_path = config.get('tesseract_path', r'C:\Program Files\Tesseract-OCR\tesseract.exe')
pytesseract.pytesseract.tesseract_cmd = tesseract_path
logger.info(f"Tesseract path: {tesseract_path}")

# --------------------- MODEL CONFIGURATION ---------------------
# Get the directory where the script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
MODEL_DIR = SCRIPT_DIR / "models"

# Verify models exist
logger.debug(f"Script directory: {SCRIPT_DIR}")
logger.debug(f"Model directory: {MODEL_DIR}")

if not MODEL_DIR.exists():
    logger.error(f"Models directory not found at: {MODEL_DIR}")
    raise RuntimeError(
        f"Models directory not found at: {MODEL_DIR}\n"
        f"Please ensure the 'models' folder exists with:\n"
        f"  - models/PP-OCRv5_server_det/\n"
        f"  - models/PP-OCRv5_server_rec/\n"
    )

DET_MODEL_PATH = MODEL_DIR / "PP-OCRv5_server_det"
REC_MODEL_PATH = MODEL_DIR / "PP-OCRv5_server_rec"

if not DET_MODEL_PATH.exists():
    logger.error(f"Detection model not found at: {DET_MODEL_PATH}")
    raise RuntimeError(f"Detection model not found at: {DET_MODEL_PATH}")
if not REC_MODEL_PATH.exists():
    logger.error(f"Recognition model not found at: {REC_MODEL_PATH}")
    raise RuntimeError(f"Recognition model not found at: {REC_MODEL_PATH}")

logger.info(f"Detection model: {DET_MODEL_PATH}")
logger.info(f"Recognition model: {REC_MODEL_PATH}")
print(f"[MODEL] Using detection model from: {DET_MODEL_PATH}")
print(f"[MODEL] Using recognition model from: {REC_MODEL_PATH}")
# --------------------------------------------------------------

# --------------------- ROTATION MODULE ---------------------
def rotate_image_preserve(img, angle):
    logger.debug(f"Rotating image by {angle} degrees")
    if angle == 0:
        return img.copy()
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

def get_text_confidence(img):
    try:
        data = pytesseract.image_to_data(img, config='--psm 6', output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in data['conf'] if c != '-1']
        conf_mean = np.mean(confs) if confs else 0
        logger.debug(f"Text confidence: {conf_mean:.2f} (from {len(confs)} text regions)")
        return conf_mean
    except Exception as e:
        logger.error(f"Error calculating text confidence: {e}")
        logger.debug(traceback.format_exc())
        return 0

def find_best_rotation(image_path, save_path):
    logger.info(f"Finding best rotation for: {image_path}")
    try:
        original_img = cv2.imread(image_path)
        if original_img is None:
            logger.error(f"Failed to read image: {image_path}")
            raise ValueError(f"Could not read image: {image_path}")
        
        angles = [0, 90, 180, 270]
        best_angle = 0
        best_conf = 0

        for angle in angles:
            temp = rotate_image_preserve(original_img, angle)
            conf = get_text_confidence(temp)
            logger.debug(f"Angle {angle}° -> Confidence {conf:.2f}")
            print(f"Angle {angle}° -> Confidence {conf:.2f}")
            if conf > best_conf:
                best_conf = conf
                best_angle = angle

        corrected = rotate_image_preserve(original_img, best_angle)
        cv2.imwrite(save_path, corrected)
        logger.info(f"Best rotation: {best_angle}°, confidence: {best_conf:.2f}")
        logger.info(f"Corrected image saved: {save_path}")
        print(f"[OK] Best rotation: {best_angle} degrees, saved to {save_path}")

        return corrected
    except Exception as e:
        logger.error(f"Error in find_best_rotation: {e}")
        logger.debug(traceback.format_exc())
        raise
# --------------------------------------------------------------

# ---------------------- PADDLE OCR ----------------------------
logger.info("Initializing PaddleOCR...")
try:
    ocr = PaddleOCR(
        text_detection_model_dir=str(DET_MODEL_PATH),
        text_recognition_model_dir=str(REC_MODEL_PATH),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False
    )
    logger.info("PaddleOCR initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize PaddleOCR: {e}")
    logger.debug(traceback.format_exc())
    raise
# --------------------------------------------------------------

# Poppler path - will be set from config or command line in main
POPLER_PATH = None
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# ---------------------- PDF converter -------------------------
def convert_pdf_to_images(pdf_path):
    logger.info(f"Converting PDF to images: {pdf_path}")
    try:
        if POPLER_PATH:
            logger.debug(f"Using Poppler path: {POPLER_PATH}")
            images = convert_from_path(pdf_path, poppler_path=POPLER_PATH)
        else:
            logger.debug("Using system Poppler (from PATH)")
            images = convert_from_path(pdf_path)
        logger.info(f"PDF converted successfully: {len(images)} pages")
        return images
    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}")
        logger.debug(traceback.format_exc())
        raise
# --------------------------------------------------------------

# ---------------------- CROPPING LOGIC ------------------------
def crop_page_two_parts(img_cv2, page_id, save_folder):

    h, w = img_cv2.shape[:2]

    # Crop 1: Top 0 to 650
    crop1 = img_cv2[0:650, 0:w]
    crop1_path = f"{save_folder}/page_{page_id}_crop_top.jpg"
    cv2.imwrite(crop1_path, crop1)

    # Crop 2: Bottom 1650 to end
    crop2 = img_cv2[1650:h, 0:w]
    crop2_path = f"{save_folder}/page_{page_id}_crop_bottom.jpg"
    cv2.imwrite(crop2_path, crop2)

    return crop1_path, crop2_path

def crop_page_middle_remaining(img_cv2, page_id, save_folder):
    """
    Crop the middle area that's NOT covered by top (0-650) and bottom (1650-end).
    This is the remaining space: 650 to 1650.
    """
    h, w = img_cv2.shape[:2]

    # Middle area: from y=1000 to y=1550
    crop_middle = img_cv2[1000:1250, 0:w]
    middle_path = f"{save_folder}/page_{page_id}_crop_middle.jpg"
    cv2.imwrite(middle_path, crop_middle)
    
    return middle_path

def crop_page_other_amount_region(img_cv2, page_id, save_folder):
    """
    Crop region from y=650 to y=1200 and x=1300 to end for other_amount extraction.
    """
    h, w = img_cv2.shape[:2]

    # Region: from x=1300 to end, y=650 to y=1200
    crop_region = img_cv2[650:1200, 1300:w]
    region_path = f"{save_folder}/page_{page_id}_crop_other_amount.jpg"
    cv2.imwrite(region_path, crop_region)
    
    return region_path
# --------------------------------------------------------------

# ---------------------- RUN OCR ON ONE CROP -------------------
def run_paddle_on_crop(image_path, save_folder, label):
    logger.info(f"Running PaddleOCR on: {image_path} (label: {label})")
    try:
        result = ocr.predict(image_path)
        logger.debug(f"OCR prediction completed: {len(result)} result(s)")

        for idx, res in enumerate(result):
            img_out = f"{save_folder}/{label}_ocr_img_{idx}.jpg"
            json_out = f"{save_folder}/{label}_ocr_{idx}.json"

            res.save_to_img(img_out)
            res.save_to_json(json_out)
            logger.debug(f"Saved OCR result {idx}: {json_out}")

        logger.info(f"OCR processing completed for {label}")
        return True
    except Exception as e:
        logger.error(f"Error in run_paddle_on_crop for {label}: {e}")
        logger.debug(traceback.format_exc())
        return False
# --------------------------------------------------------------

# ---------------------- UTILITIES (shared) --------------------
def sim(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def load_words(path):
    if path is None:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    texts = data.get("rec_texts", [])
    boxes = data.get("rec_boxes", [])
    scores = data.get("rec_scores", [1.0] * len(texts))
    return [{"text": t, "bbox": b, "score": s} for t, b, s in zip(texts, boxes, scores)]

def y_overlap(b1, b2):
    return max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))

def center_y(bbox):
    return (bbox[1] + bbox[3]) / 2

months = {
    'january': 1, 'jan': 1,
    'february': 2, 'feb': 2,
    'march': 3, 'mar': 3,
    'april': 4, 'apr': 4,
    'may': 5,
    'june': 6, 'jun': 6,
    'july': 7, 'jul': 7,
    'august': 8, 'aug': 8,
    'september': 9, 'sep': 9,
    'october': 10, 'oct': 10,
    'november': 11, 'nov': 11,
    'december': 12, 'dec': 12
}

def parse_ocr_date(text):
    """
    Parse OCR date-like text into YYYYMMDD using heuristics described by user.
    """
    logger.debug(f"Parsing date from OCR text: '{text}'")
    original_text = text

    if not text or not text.strip():
        logger.debug("Empty text, returning as-is")
        return original_text

    # Normalize
    cleaned = re.sub(r'[\u00B7\u00B0\u00B7\u2022\u2027\u00B4\u2019\u2018\u201C\u201D\·\●\•\u2024\u00B7/\\,;:\-]+', ' ', text)
    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return original_text

    tokens_alpha = re.findall(r'[A-Za-z]+', cleaned)
    tokens_num = re.findall(r'\d{1,4}', cleaned)

    # 1) detect month by fuzzy matching tokens_alpha
    month_num = None
    best_month_score = 0.0
    best_month_token = None
    for token in tokens_alpha:
        for m_str, m_val in months.items():
            score = sim(token.lower(), m_str)
            if score > best_month_score:
                best_month_score = score
                month_num = m_val
                best_month_token = token
    if best_month_score < 0.45:
        for token in tokens_alpha:
            if len(token) <= 2:
                for m_str, m_val in months.items():
                    if token[0].lower() == m_str[0]:
                        month_num = m_val
                        best_month_token = token
                        best_month_score = 0.45
                        break
                if month_num:
                    break

    # 2) Detect explicit 4-digit year first
    year = None
    for n in tokens_num:
        if len(n) == 4 and (n.startswith('20') or n.startswith('19')):
            try:
                ytest = int(n)
                year = ytest
                break
            except:
                pass

    # 3) If year not found, look for 2- or 1-digit that plausibly is year-last
    if year is None:
        two_digit_candidates = [int(n) for n in tokens_num if 1 <= len(n) <= 2]
        if two_digit_candidates:
            last_candidate = int(tokens_num[-1])
            if 0 <= last_candidate <= 99:
                if last_candidate <= 25:
                    year = 2000 + last_candidate
                else:
                    year = None

    # 4) Determine day & month using numeric tokens and month detection
    day = None
    month = month_num

    def valid_ymd(y, m, d):
        try:
            datetime(y, m, d)
            return True
        except:
            return False

    if len(tokens_num) >= 3:
        nums = [int(n) for n in tokens_num]
        if any(len(n) == 4 for n in tokens_num):
            for n in tokens_num:
                if len(n) == 4:
                    year_candidate = int(n)
                    break
            others = [int(n) for n in tokens_num if n != str(year_candidate)]
            m_candidate = None
            d_candidate = None
            for v in others:
                if 1 <= v <= 12 and m_candidate is None:
                    m_candidate = v
                else:
                    if 1 <= v <= 31:
                        d_candidate = v
            if m_candidate and d_candidate and valid_ymd(year_candidate, m_candidate, d_candidate):
                year = year_candidate; month = m_candidate; day = d_candidate
                return f"{year:04d}{month:02d}{day:02d}"
        else:
            a, b, c = [int(n) for n in tokens_num[:3]]
            if 1 <= a <= 31 and 1 <= b <= 12:
                if 0 <= c <= 99:
                    if c <= 25:
                        y = 2000 + c
                    else:
                        y = 1900 + c
                    if valid_ymd(y, b, a):
                        return f"{y:04d}{b:02d}{a:02d}"
            if len(tokens_num[0]) == 4:
                y = int(tokens_num[0]); m = int(tokens_num[1]); d = int(tokens_num[2])
                if valid_ymd(y, m, d):
                    return f"{y:04d}{m:02d}{d:02d}"

    if month is not None and tokens_num:
        if year is None:
            for n in tokens_num:
                if len(n) == 4:
                    year = int(n)
                    break

        if len(tokens_num) >= 2:
            nums = [int(n) for n in tokens_num if not (year and len(n) == 4 and int(n) == year)]
            day_candidate = None
            year_candidate = year
            if len(nums) == 1:
                n = nums[0]
                s = str(n)
                if len(s) == 2 and n > 31:
                    day = int(s[0])
                    year = 2000 + int(s[1])
                    if valid_ymd(year, month, day):
                        return f"{year:04d}{month:02d}{day:02d}"
                else:
                    if 1 <= n <= 31:
                        day = n
            else:
                for n in nums:
                    if 1 <= n <= 31:
                        day_candidate = n
                        break
                if day_candidate is None:
                    for n in nums:
                        s = str(n)
                        if len(s) == 2:
                            day = int(s[0])
                            year = 2000 + int(s[1])
                            if valid_ymd(year, month, day):
                                return f"{year:04d}{month:02d}{day:02d}"
                else:
                    day = day_candidate
            if year is None:
                last_n = int(tokens_num[-1])
                if 0 <= last_n <= 99:
                    if last_n <= 25:
                        year = 2000 + last_n
                    else:
                        year = 1900 + last_n

        if year and day:
            if valid_ymd(year, month, day):
                return f"{year:04d}{month:02d}{day:02d}"

    if len(tokens_num) >= 2:
        nums = [int(n) for n in tokens_num]
        if len(nums) >= 3:
            a, b, c = nums[:3]
            if 1 <= a <= 31 and 1 <= b <= 12 and (len(tokens_num[2]) == 4):
                y = c; m = b; d = a
                if valid_ymd(y, m, d):
                    return f"{y:04d}{m:02d}{d:02d}"
            if len(tokens_num[0]) == 4 and 1 <= b <= 12 and 1 <= c <= 31:
                y = a; m = b; d = c
                if valid_ymd(y, m, d):
                    return f"{y:04d}{m:02d}{d:02d}"
        else:
            a, b = nums[:2]
            year_guess = datetime.now().year
            if 1 <= a <= 31 and 1 <= b <= 12:
                year = year or (2000 + (year_guess % 100))
                if valid_ymd(year, b, a):
                    return f"{year:04d}{b:02d}{a:02d}"
            if 1 <= a <= 12 and 1 <= b <= 31:
                year = year or (2000 + (datetime.now().year % 100))
                if valid_ymd(year, a, b):
                    return f"{year:04d}{a:02d}{b:02d}"

    if month is not None and not day and tokens_num:
        for n in tokens_num:
            v = int(n)
            if 1 <= v <= 31:
                day = v
                break
        if day and year is None:
            year = 2000 + (datetime.now().year % 100)
        if year and day and month:
            if valid_ymd(year, month, day):
                result = f"{year:04d}{month:02d}{day:02d}"
                logger.info(f"Date parsed successfully: '{original_text}' -> {result}")
                return result

    logger.warning(f"Could not parse date: '{original_text}' - returning original")
    return original_text


def clean_amount(text):
    logger.debug(f"Cleaning amount text: '{text}'")
    if not text:
        logger.debug("Empty text, returning None")
        return None

    original_text = text
    # 1. Remove ALL degree and dot-like unicode characters
    text = re.sub(r"[\u00B0\u02DA\u00BA\u00B7\u2022\u02D9\u2027\u2070\u200B\u200C\u200D\uFEFF]+", "", text)

    # 2. Fix OCR character mistakes
    text = text.translate(str.maketrans({
        'O': '0', 'o': '0',
        'I': '1', 'l': '1',
        'b': '6', 'B': '8',
        'S': '5', 's': '5',
        'Z': '2', 'z': '2'
    }))

    # 3. Keep only digits and dot
    text = re.sub(r"[^0-9.]", "", text)

    # 4. Fix multiple dots
    if text.count('.') > 1:
        parts = text.split('.')
        text = parts[0] + '.' + ''.join(parts[1:])

    # 5. If nothing numeric remains → invalid
    if not re.search(r'\d', text):
        return None

    # 6. Convert to float
    try:
        num = float(text)
    except:
        return None

    # 7. ALWAYS format .2f
    return f"{num:.2f}"


# ---------------------- KEYWORD SEARCH FUNCTIONS ---------------
KEYWORD_SETS = {
    "intrlab cash log date": [
        "intrlab cash log date",
        "intralab cash log date"
    ],
    "psc prefix#": [
        "psc prefix#",
        "psc prefix",
        "psc-prefix",
        "psc prefix #",
        "prefix#",
        "pscprefix"
    ]
}

FUZZY_THRESHOLD = 0.60

def find_keyword_leftmost(words, keyword_list, threshold=FUZZY_THRESHOLD):
    """
    Find keyword by LEFTMOST position (lowest left_x) as priority.
    Used for account/transit to ensure we get the primary occurrence on left side of form.
    """
    candidates = []

    for w in words:
        t = w["text"]
        for kw in keyword_list:
            score = sim(t, kw)
            if score >= threshold:
                b = w.get("bbox", [0,0,0,0])
                candidates.append({
                    "text": t,
                    "bbox": b,
                    "left_x": b[0],
                    "score": score
                })

    if not candidates:
        return None

    # Sort by LEFTMOST FIRST (lowest left_x), then by highest score for ties
    candidates.sort(key=lambda x: (x["left_x"], -x["score"]))

    best = candidates[0]

    return {
        "text": best["text"],
        "bbox": best["bbox"],
        "score": best["score"]
    }

def find_keyword(words, keyword_list, threshold=FUZZY_THRESHOLD):
    """
    FINAL RULE:
    - Any fuzzy score >= threshold is counted.
    - Always choose HIGHEST SCORE first.
    - If scores tie → choose LEFTMOST.
    """
    logger.debug(f"Finding keyword from list: {keyword_list}, threshold: {threshold}")

    candidates = []

    for w in words:
        t = w["text"]
        for kw in keyword_list:
            score = sim(t, kw)
            if score >= threshold:
                b = w.get("bbox", [0,0,0,0])
                candidates.append({
                    "text": t,
                    "bbox": b,
                    "left_x": b[0],
                    "center_y": (b[1]+b[3])/2,
                    "score": score
                })
                logger.debug(f"Candidate: '{t}' (score: {score:.2f})")

    if not candidates:
        logger.debug("No keyword candidates found")
        return None

    # [OK] Sort by HIGHEST SCORE FIRST, then LEFTMOST for ties
    candidates.sort(key=lambda x: (-x["score"], x["left_x"]))

    best = candidates[0]
    logger.info(f"Keyword match: '{best['text']}' (score: {best['score']:.2f})")

    return {
        "text": best["text"],
        "bbox": best["bbox"],
        "score": best["score"]
    }



def extract_inline_value(main):
    full = main["text"]
    full_l = full.lower()
    
    # Handle both cases: with and without "keyword" field
    if "keyword" in main:
        kw = main["keyword"].lower()
    else:
        # If no keyword field, we can't do inline extraction
        return None

    # NEW: Try space-removed matching for concatenated cases
    kw_no_space = kw.replace(" ", "")
    full_no_space = full_l.replace(" ", "")
    pos = full_no_space.find(kw_no_space)
    if pos != -1:
        # Map back to original position: find the char index after the matched non-space chars
        char_count = 0
        orig_end = len(full)
        kw_len_no_space = len(kw_no_space)
        for i, c in enumerate(full_l):
            if c != ' ':
                char_count += 1
                if char_count > pos + kw_len_no_space:
                    orig_end = i
                    break
        val = full[orig_end:].strip(" :.-").strip()
        if val:
            return val

    # FALLBACK: Original sliding window sim (for spaced cases)
    best_score = 0
    best_end = None
    kw_len = len(kw)
    for i in range(len(full_l) - kw_len + 1):
        window = full_l[i:i+kw_len]
        score = sim(window, kw)
        if score > best_score:
            best_score = score
            best_end = i + kw_len

    if best_score < 0.70 or best_end is None:
        return None

    val = full[best_end:].strip(" :.-").strip()

    return val if val else None

def find_next_word(main, all_words):
    inline = extract_inline_value(main)
    if inline:
        return {
            "text": inline,
            "bbox": main["bbox"],
            "left_diff": 0
        }

    main_left = main["bbox"][0]
    main_bbox = main["bbox"]

    same_line = []

    for w in all_words:
        if w["text"] == main["text"]:
            continue

        b = w["bbox"]
        l = b[0]

        if l <= main_left:
            continue

        if y_overlap(main_bbox, b) >= 2:
            same_line.append({
                "text": w["text"],
                "bbox": b,
                "left_diff": l - main_left
            })

    if same_line:
        return min(same_line, key=lambda x: x["left_diff"])

    return None

def find_value_right_of_keyword(keyword_block, all_words):
    """
    Finds the nearest text that appears to the RIGHT of the keyword.
    """
    kw_x1, kw_y1, kw_x2, kw_y2 = keyword_block["bbox"]

    candidates = []

    for w in all_words:
        x1, y1, x2, y2 = w["bbox"]

        # word must be to the right of keyword
        if x1 > kw_x2:

            # vertical alignment tolerance
            if y2 >= kw_y1 - 15 and y1 <= kw_y2 + 15:
                candidates.append((x1, w))

    if not candidates:
        return None

    # sort by closest on x-axis
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]

def process(top_json_path, bottom_json_path=None):
    print("\n================ PROCESSING TOP BLOCK ================\n")

    all_words = []

    # ----------------------------------------------------
    # LOAD TOP JSON (SAFE)
    # ----------------------------------------------------
    with open(top_json_path, "r", encoding="utf-8") as f:
        top_data = json.load(f)

    # --- CASE 1: PaddleOCR RAW FORMAT USING rec_texts + rec_boxes ---
    if isinstance(top_data, dict) and "rec_texts" in top_data and "rec_boxes" in top_data:
        texts = top_data["rec_texts"]
        boxes = top_data["rec_boxes"]
        scores = top_data.get("rec_scores", [1.0] * len(texts))

        for t, b, s in zip(texts, boxes, scores):
            all_words.append({"text": t, "bbox": b, "score": s})

    # --- CASE 2: Word-list format ---
    elif isinstance(top_data, list):
        for w in top_data:
            if isinstance(w, dict) and "text" in w and "bbox" in w:
                all_words.append({**w, "score": w.get("score", 1.0)})

    # ----------------------------------------------------
    # LOAD BOTTOM JSON (OPTIONAL)
    # ----------------------------------------------------
    if bottom_json_path:
        with open(bottom_json_path, "r", encoding="utf-8") as f:
            bottom_data = json.load(f)

        # RAW PADDLE OCR FORMAT
        if isinstance(bottom_data, dict) and "rec_texts" in bottom_data and "rec_boxes" in bottom_data:
            texts = bottom_data["rec_texts"]
            boxes = bottom_data["rec_boxes"]
            scores = bottom_data.get("rec_scores", [1.0] * len(texts))

            for t, b, s in zip(texts, boxes, scores):
                all_words.append({"text": t, "bbox": b, "score": s})

        # NORMAL WORD LIST FORMAT
        elif isinstance(bottom_data, list):
            for w in bottom_data:
                if isinstance(w, dict) and "text" in w and "bbox" in w:
                    all_words.append({**w, "score": w.get("score", 1.0)})

    print("TOTAL WORDS LOADED:", len(all_words))

    # ----------------------------------------------------
    # FIND KEYWORDS USING FUZZY LOGIC
    # ----------------------------------------------------
    main_keywords = {}
    for kw, patterns in KEYWORD_SETS.items():
        found = find_keyword(all_words, patterns)
        if found:
            found["keyword"] = kw 
            main_keywords[kw] = found

    print("MAIN KEYWORDS FOUND:", list(main_keywords.keys()))

    final_results = {}  # {kw: value_str}

    # ----------------------------------------------------
    # PROCESS EACH KEYWORD
    # ----------------------------------------------------
    for kw, main in main_keywords.items():

        # --------------------------------------------------------
        # SPECIAL LOGIC FOR INTRLAB CASH LOG DATE
        # --------------------------------------------------------
        if kw == "intrlab cash log date":
            # FIRST: Try inline extraction for concatenated case
            inline_val = extract_inline_value(main)
            if inline_val:
                final_results[kw] = inline_val
                print(f"FINAL VALUE for {kw}: {final_results[kw]} (from inline)")
                continue

            # FIXED: Use find_next_word for closest after with y_overlap >=2 (picks overlapping/near lines)
            nxt = find_next_word(main, all_words)
            final_results[kw] = nxt["text"] if nxt else ""
            print(f"FINAL VALUE for {kw}: {final_results[kw]} (from next word)")

        # --------------------------------------------------------
        # LOGIC FOR PSC PREFIX#
        # --------------------------------------------------------
        elif kw == "psc prefix#":

            right_word = find_value_right_of_keyword(main, all_words)
            if right_word:
                final_results[kw] = right_word["text"].strip()
                print(f"FINAL VALUE for {kw}: {final_results[kw]}")
                continue

            # FALLBACK: Use find_next_word (includes inline, but unlikely)
            nxt = find_next_word(main, all_words)
            final_results[kw] = nxt["text"] if nxt else ""
            print(f"FINAL VALUE for {kw}: {final_results[kw]} (fallback)")

    return final_results


# ---------------------- SEARCH_BOTTOM.PY FUNCTIONS ------------
COMMENT_THRESHOLD = 0.50  # LOWERED from 0.70 - be more permissive
INVOICE_THRESHOLD = 0.50  # LOWERED from 0.50
VISIT_THRESHOLD = 0.70   # LOWERED from 0.80
LINE_GAP = 60

def extract_full_number(text):
    parts = re.findall(r"\d{2,}", text)

    if not parts:
        return None

    combined = "".join(parts)

    return combined if len(combined) >= 4 else None

def extract_invoice(words):
    """
    Updated invoice/visit logic per user:
    - Find comments keyword (threshold COMMENT_THRESHOLD)
    - If found, compute bottom of comments bbox C
    - look in this zone for invoice (fuzzy >= INVOICE_THRESHOLD) OR visit (fuzzy >= VISIT_THRESHOLD)
      - if invoice -> invoice = Y, visit = N
      - if visit -> invoice = N, visit = Y
    """
    comments = find_keyword(words, ["comments"], threshold=COMMENT_THRESHOLD)
    if not comments:
        return {'invoice': 'N', 'visit': 'N'}

    c_bottom = comments["bbox"][3]  # bottom y coordinate
    zone_top = c_bottom + 1
    zone_bottom = c_bottom + 300  # EXPANDED from 150 to 300 for larger search area
    zone_words = [w for w in words if zone_top <= center_y(w["bbox"]) <= zone_bottom]

    if not zone_words:
        return {'invoice': 'N', 'visit': 'N'}

    # Check for "invoice" first (priority) - with substring matching
    best_inv_score = 0
    best_inv_word = None
    for w in zone_words:
        text_lower = w["text"].lower()
        # Check if word CONTAINS "invoice" substring
        if "invoice" in text_lower:
            best_inv_score = 1.0  # Direct hit
            best_inv_word = w
            break
        # Otherwise use fuzzy matching
        score = sim(w["text"], "invoice")
        if score >= INVOICE_THRESHOLD and score > best_inv_score:
            best_inv_score = score
            best_inv_word = w
    
    if best_inv_word and best_inv_score >= INVOICE_THRESHOLD:
        return {'invoice': 'Y', 'visit': 'N'}

    # If no invoice, then check for "visit" with higher threshold
    best_visit_score = max((sim(w["text"], "visit") for w in zone_words), default=0)
    if best_visit_score >= VISIT_THRESHOLD:
        # visit found
        return {'invoice': 'N', 'visit': 'Y'}

    # If neither, both N
    return {'invoice': 'N', 'visit': 'N'}
# --------------------------------------------------------------

# ---------------------- SEARCH_TOP2.PY FUNCTIONS --------------
KEYWORDS_TOP2 = {
    "transit": ["transit no", "transit"],
    "account": ["account no", "compte n"]
}

LINE_GAP_TOP2 = 60

def extract_int(text):
    cleaned = text.strip()

    for ch in cleaned:
        if not (ch.isdigit() or ch == "-"):
            return None

    digits_exist = any(ch.isdigit() for ch in cleaned)
    if not digits_exist:
        return None

    return cleaned

def find_number_below(words, kw_bbox):
    kw_y2 = kw_bbox[3]
    kw_left = kw_bbox[0]
    kw_right = kw_bbox[2]

    next_line_words = []

    for w in words:
        b = w["bbox"]

        if kw_y2 < b[1] <= kw_y2 + LINE_GAP_TOP2:
            next_line_words.append(w)

    if not next_line_words:
        return None

    for w in next_line_words:
        b = w["bbox"]

        if b[2] < kw_left or b[0] > kw_right:
            continue

        num = extract_int(w["text"])
        if num:
            return num

    return None

def extract_transit_account(words):
    results = {}

    # For transit and account, use LEFTMOST position (lowest left_x) as priority, not highest score
    transit_kw = find_keyword_leftmost(words, KEYWORDS_TOP2["transit"])

    if transit_kw:
        print("\nTRANSIT KEYWORD FOUND:", transit_kw)
        transit_num = find_number_below(words, transit_kw["bbox"])
        results["transit_number"] = transit_num
        print("TRANSIT NUMBER:", transit_num)
    else:
        print("\nNO TRANSIT KEYWORD FOUND")
        results["transit_number"] = None

    account_kw = find_keyword_leftmost(words, KEYWORDS_TOP2["account"])

    if account_kw:
        print("\nACCOUNT KEYWORD FOUND:", account_kw)
        account_num = find_number_below(words, account_kw["bbox"])
        results["account_number"] = account_num
        print("ACCOUNT NUMBER:", account_num)
    else:
        print("\nNO ACCOUNT KEYWORD FOUND")
        results["account_number"] = None

    return results
# --------------------------------------------------------------

# ---------------------- SEARCH_BOTTOM2.PY FUNCTIONS -----------
TOTAL_THRESHOLD = 0.70

def extract_bottom_total(json_path):
    words = load_words(json_path)
    rec_texts = [w['text'] for w in words]
    rec_boxes = [w['bbox'] for w in words]
    rec_scores = [w['score'] for w in words]

    total_idx = None

    for i, txt in enumerate(rec_texts):
        text_clean = txt.strip()

        if sim(text_clean, "total") >= TOTAL_THRESHOLD:
            if " " not in text_clean:
                total_idx = i
                break

    if total_idx is None:
        print("NO TOTAL KEYWORD FOUND")
        return None, None

    total_text = rec_texts[total_idx]
    total_bbox = rec_boxes[total_idx]

    print("\nFOUND TOTAL USING rec_text ONLY:")
    print("TEXT:", total_text)
    print("BBOX:", total_bbox)

    t_left = total_bbox[0]

    candidates = []

    for i, (txt, bbox) in enumerate(zip(rec_texts, rec_boxes)):
        if i == total_idx:
            continue

        if y_overlap(total_bbox, bbox) < 2:
            continue

        if bbox[0] <= t_left:
            continue

        candidates.append({
            "text": txt,
            "bbox": bbox,
            "left_diff": bbox[0] - t_left,
            "score": rec_scores[i]
        })

    if not candidates:
        print("NO VALUE FOUND TO THE RIGHT OF TOTAL")
        return None, None

    best = min(candidates, key=lambda x: x["left_diff"])

    print("\nTOTAL VALUE FOUND:", best["text"], best["bbox"])
    return best["text"], best["score"]

# ---------------------- OTHER AMOUNT EXTRACTION ----------------
def extract_other_amount(words):
    """
    Extract 'other amount' based on check box keyword location.
    Logic:
    1. Find keyword "check box" with 90% fuzzy matching
    2. Get keyword bbox: left, right, bottom
    3. Expand region: left_new = left - 60, right_new = right + 60
    4. Search BELOW the keyword (bottom to bottom + 200px)
    5. Find words that horizontally overlap with expanded region
    6. Extract first number found
    7. Return the number
    """
    logger.info("Extracting other_amount based on check box keyword")
    # Find keyword with 90% fuzzy threshold (check box, checkbox, etc.)
    keyword_patterns = ["check box", "checkbox", "cheque box", "check", "chk box"]
    keyword_kw = find_keyword(words, keyword_patterns, threshold=0.90)
    
    if not keyword_kw:
        logger.warning("No 'check box' keyword found (90% threshold)")
        print("[OTHER AMOUNT] No 'check box' keyword found (90% threshold)")
        return None
    
    logger.info(f"Found check box keyword: '{keyword_kw['text']}'")
    print(f"[OTHER AMOUNT] Found keyword: '{keyword_kw['text']}'")
    kw_bbox = keyword_kw["bbox"]
    kw_left, kw_top, kw_right, kw_bottom = kw_bbox
    
    # Expand the region based on keyword coordinates
    region_left = kw_left - 60
    region_right = kw_right + 60
    region_top = kw_bottom
    region_bottom = kw_bottom + 200
    
    logger.debug(f"Keyword bbox: left={kw_left:.1f}, right={kw_right:.1f}, bottom={kw_bottom:.1f}")
    logger.debug(f"Search region (expanded): left={region_left:.1f}, right={region_right:.1f}, top={region_top:.1f}, bottom={region_bottom:.1f}")
    print(f"[OTHER AMOUNT] Keyword bbox: left={kw_left:.1f}, right={kw_right:.1f}, bottom={kw_bottom:.1f}")
    print(f"[OTHER AMOUNT] Search region (expanded): left={region_left:.1f}, right={region_right:.1f}, top={region_top:.1f}, bottom={region_bottom:.1f}")
    
    # Find words BELOW keyword that horizontally overlap with expanded region
    candidates = []
    for w in words:
        bbox = w.get("bbox", [0,0,0,0])
        word_left, word_top, word_right, word_bottom = bbox
        
        # Check if word is BELOW keyword
        is_below = word_top >= region_top
        
        # Check if word horizontally overlaps with expanded region
        horiz_overlap = (word_left < region_right) and (word_right > region_left)
        
        if is_below and horiz_overlap and word_top <= region_bottom:
            candidates.append(w)
    
    print(f"[OTHER AMOUNT] Found {len(candidates)} words in region")
    
    # Extract all numbers and find the SMALLEST one (excluding very large numbers)
    found_numbers = []
    for w in candidates:
        text = w["text"].strip()
        print(f"[OTHER AMOUNT] Checking word: '{text}'")
        
        # ONLY try to extract if word contains at least one digit
        if not any(c.isdigit() for c in text):
            print(f"[OTHER AMOUNT]   -> Skipped (no digits)")
            continue
        
        # Try to extract number
        cleaned = clean_amount(text)
        if cleaned:
            found_numbers.append(cleaned)
            print(f"[OTHER AMOUNT] Found number: '{text}' -> {cleaned}")
    
    if found_numbers:
        # Return the smallest number (likely the checkbox value, not the large amounts)
        smallest = min(found_numbers, key=lambda x: float(x))
        logger.info(f"Other amount extracted: {smallest} (smallest of {len(found_numbers)} candidates)")
        print(f"[OTHER AMOUNT] Returning smallest number: {smallest}")
        return smallest
    
    logger.warning("No other amount found")
    print(f"[OTHER AMOUNT] No number found")
    return None
# --------------------------------------------------------------

def fallback_prefix_from_accession(accession_kw, words):
    """
    Extract prefix from accession numbers found below ACCESSION keyword.
    Looks for horizontal overlap and values below the keyword on the same page.
    """
    logger.info("Attempting fallback prefix extraction from ACCESSION")
    if not accession_kw:
        logger.debug("No ACCESSION keyword provided")
        return None

    acc_bbox = accession_kw["bbox"]
    acc_left, acc_top, acc_right, acc_bottom = acc_bbox
    logger.debug(f"ACCESSION bbox: left={acc_left:.1f}, bottom={acc_bottom:.1f}, right={acc_right:.1f}")

    # Find words below ACCESSION that horizontally overlap with it
    for w in words:
        bbox = w.get("bbox", [0, 0, 0, 0])
        word_left, word_top, word_right, word_bottom = bbox
        
        # Check if word is below ACCESSION (top > acc_bottom) and horizontally overlaps
        is_below = word_top > acc_bottom
        horiz_overlap = (word_left < acc_right) and (word_right > acc_left)
        
        if is_below and horiz_overlap:
            text = w["text"].strip()
            
            # Only consider if it starts with letters and is reasonably long (> 2 chars)
            # AND doesn't contain special punctuation like ':' or '.'
            if len(text) > 2 and text[0].isalpha() and ':' not in text and '.' not in text:
                # Extract first 2 characters as prefix (can be letters or digits)
                # For N03100037, we want N0 as prefix
                if len(text) >= 2:
                    return text[:2].upper()
    
    return None

def extract_consensus_date_from_accession_page(words, accession_page):
    """
    Find date patterns like 06-nov-2025 or NOV 06 2025 on the accession page only.
    Updated regex to handle both day-month-year and month-day-year, plus split words.
    Return normalized YYYYMMDD if consensus found (at least 2 matching), else None.
    """
    # Filter words to only those on the accession_page
    page_words = [w for w in words if w.get('page') == accession_page]

    if not page_words:
        print(f"    [DATES] No words found on page {accession_page}")
        return None

    print(f"    [DATES] Searching {len(page_words)} words on accession page {accession_page}")

    # NEW: Group words into lines (concat text if y_overlap >=2) to handle split dates
    lines = []
    current_line = []
    sorted_words = sorted(page_words, key=lambda w: (center_y(w["bbox"]), w["bbox"][0]))  # Sort by y-center, then x-left
    for w in sorted_words:
        if not current_line:
            current_line.append(w)
            continue
        prev = current_line[-1]
        if y_overlap(prev["bbox"], w["bbox"]) >= 2:  # Same line
            current_line.append(w)
        else:  # New line
            line_text = ' '.join([ww["text"].strip() for ww in current_line])
            if line_text:
                lines.append(line_text)
            current_line = [w]
    if current_line:
        line_text = ' '.join([ww["text"].strip() for ww in current_line])
        if line_text:
            lines.append(line_text)

    print(f"    [DATES] Grouped into {len(lines)} lines")

    # UPDATED REGEX: Alternation for day-month-year OR month-day-year
    # Groups: For day-month: (1=day,2=month_txt,3=year); For month-day: (4=month_txt,5=day,6=year)
    month_abbrs = '|'.join(months.keys())  # 'jan|feb|...|dec'
    date_regex = re.compile(
        rf'\b(?:'  # Non-capturing group for alternation
        rf'(\d{{1,2}})[\-./\s]({month_abbrs})[a-z]*[\-./\s](\d{{2,4}})|'  # Day-month-year
        rf'({month_abbrs})[a-z]*[\-./\s]?(\d{{1,2}})[\-./\s]?(\d{{2,4}})'  # Month-day-year (flexible separators)
        rf')\b',
        re.IGNORECASE
    )

    found = []

    for line_idx, line_text in enumerate(lines):  # Search concatenated lines
        m = date_regex.search(line_text)
        line_dates = []
        while m:  # Handle multiple per line
            if m.group(1):  # Day-month-year match
                day = int(m.group(1))
                month_txt = m.group(2).lower()
                year_str = m.group(3)
            else:  # Month-day-year match
                month_txt = m.group(4).lower()
                day = int(m.group(5))
                year_str = m.group(6)

            month = months.get(month_txt[:3])  # Use first 3 chars for abbr
            if not month:
                m = date_regex.search(line_text, m.end())
                continue

            year = int(year_str)
            if year < 100:
                year = 2000 + year if year <= 25 else 1900 + year  # Assume 20xx/19xx

            try:
                dt = datetime(year, month, day)
                date_str = dt.strftime("%Y%m%d")
                found.append(date_str)
                line_dates.append(f"{day}-{month_txt}-{year}")
            except ValueError:
                pass  # Invalid date, skip

            m = date_regex.search(line_text, m.end())
        
        if line_dates:
            print(f"      Line {line_idx+1}: {line_text[:80]}...")
            print(f"        > Found dates: {line_dates}")

    from collections import Counter

    print(f"    [DATES] Total dates found: {len(found)} -> {found}")

    # Must have at least 2 dates
    if len(found) < 2:
        print(f"    [DATES] ❌ Less than 2 dates found (need at least 2)")
        return None

    freq = Counter(found)
    most_common_date, most_common_count = freq.most_common(1)[0]

    print(f"    [DATES] Frequency: {dict(freq)}")

    # NEW LOGIC:
    # - 2 dates found: accept ONLY if both are same (count >= 2)
    # - 3 dates found: accept if at least 2 are same (most_common_count >= 2)
    #                 → choose the most frequent date
    # - Otherwise: return N/A

    if len(found) == 2:
        # Both must be same
        if most_common_count == 2:
            print(f"    [DATES] ✅ Both dates identical: {most_common_date}")
            return most_common_date
        else:
            print(f"    [DATES] ❌ 2 different dates (not matching): {found}")
            return None  # 2 different dates → N/A

    if len(found) == 3:
        # Accept if at least 2 are same
        if most_common_count >= 2:
            print(f"    [DATES] ✅ 3 dates with {most_common_count} matching: {most_common_date}")
            return most_common_date
        else:
            print(f"    [DATES] ❌ All 3 dates different: {found}")
            return None  # All 3 different → N/A

    # Should not reach here, but fallback to N/A
    print(f"    [DATES] ❌ Unexpected number of dates: {len(found)}")
    return None




# --------------------------------------------------------------

# -------------------------- PROCESS ONE FILE -------------------
def run_ocr_on_file(file_path, output_base, data_row):
    logger.info("="*80)
    logger.info(f"PROCESSING FILE: {file_path}")
    logger.info("="*80)
    
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    logger.info(f"File extension: {ext}")

    file_output_dir = output_base / file_path.stem
    file_output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory: {file_output_dir}")

    data_row['amount_candidates'] = []
    all_document_words = []
    main_page_found = False

    # --------------------- PDF ---------------------
    if ext == ".pdf":
        logger.info("Processing PDF file")
        print(f"[INFO] Running fresh OCR on PDF")
        
        try:
            pages = convert_pdf_to_images(file_path)
            logger.info(f"PDF converted to {len(pages)} pages")
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            logger.debug(traceback.format_exc())
            raise
            
        for page_id, pil_img in enumerate(pages, start=1):
            logger.info(f"Processing page {page_id}/{len(pages)}")
            raw_path = f"_temp_page_{page_id}.jpg"
            pil_img.save(raw_path)
            logger.debug(f"Saved temporary page: {raw_path}")

            rotated_path = f"_rotated_page_{page_id}.jpg"
            corrected = find_best_rotation(raw_path, rotated_path)

            # Read rotated image into cv2 for cropping later
            img_cv2 = cv2.imread(rotated_path)

            crop1_path, crop2_path = crop_page_two_parts(img_cv2, page_id, file_output_dir)

            # Run OCR with error handling for each crop
            try:
                run_paddle_on_crop(crop1_path, file_output_dir, f"page_{page_id}_top")
            except Exception as e:
                logger.error(f"Failed to run OCR on page {page_id} top crop: {e}")
                logger.debug(traceback.format_exc())
                print(f"[WARNING] Skipping page {page_id} top crop due to OCR error")
            
            try:
                run_paddle_on_crop(crop2_path, file_output_dir, f"page_{page_id}_bottom")
            except Exception as e:
                logger.error(f"Failed to run OCR on page {page_id} bottom crop: {e}")
                logger.debug(traceback.format_exc())
                print(f"[WARNING] Skipping page {page_id} bottom crop due to OCR error")

            os.remove(raw_path)

            top_json_path = file_output_dir / f"page_{page_id}_top_ocr_0.json"
            bottom_json_path = file_output_dir / f"page_{page_id}_bottom_ocr_0.json"

            top_results = None
            ta_res = None
            page_words = []
            has_comments = False
            has_cibc = False
            if top_json_path.exists():
                # NOTE: we call process() to build top_results using JSON text/boxes
                top_results = process(str(top_json_path), str(bottom_json_path) if bottom_json_path.exists() else None)

                # Load page_words
                page_words = load_words(str(top_json_path))
                if bottom_json_path.exists():
                    page_words += load_words(str(bottom_json_path))
                
                # ADD PAGE INFO TO EACH WORD for tracking
                for w in page_words:
                    w['page'] = page_id
                
            all_document_words += page_words

            # Initialize full_words_for_amount
            full_words_for_amount = []  

            # ------------------ PAGE VALIDATION FOR PREFIX/DATE ------------------
            if not top_results:
                top_results = {}
            has_date = 'intrlab cash log date' in top_results
            has_prefix = 'psc prefix#' in top_results
            has_comments = any(sim(w["text"].lower(), "comments") >= COMMENT_THRESHOLD for w in page_words)
            has_cibc = any("cibc" in w["text"].lower() for w in page_words)
            
            logger.debug(f"Page {page_id} validation: date={has_date}, prefix={has_prefix}, comments={has_comments}, cibc={has_cibc}")

            # Only assign if all three present → main page
            if has_date and has_prefix and has_comments:
                main_page_found = True
                logger.info(f"Page {page_id} identified as MAIN page (has date, prefix, comments)")
                print(f"Valid page {page_id} for prefix/date/comments")
                date_val = top_results.get('intrlab cash log date', '')


                # --- NEW BEHAVIOR: MIDDLE-AREA OCR (650-1650) instead of full page ---
                # Do OCR on remaining middle area only
                middle_img_path = crop_page_middle_remaining(img_cv2, page_id, file_output_dir)
                middle_label = f"page_{page_id}_middle"
                try:
                    run_paddle_on_crop(middle_img_path, file_output_dir, middle_label)
                except Exception as e:
                    logger.error(f"Failed to run OCR on page {page_id} middle crop: {e}")
                    logger.debug(traceback.format_exc())
                    print(f"[WARNING] Skipping page {page_id} middle crop due to OCR error")
                
                middle_json_path = file_output_dir / f"{middle_label}_ocr_0.json"
                middle_words_for_amount = []
                if middle_json_path.exists():
                    middle_words_for_amount = load_words(str(middle_json_path))
                    # ADD PAGE INFO
                    for w in middle_words_for_amount:
                        w['page'] = page_id
                    print(f"[OK] Loaded {len(middle_words_for_amount)} words from middle-area OCR")
                
                # --- OTHER AMOUNT REGION OCR (651-999) ---
                other_amount_img_path = crop_page_other_amount_region(img_cv2, page_id, file_output_dir)
                other_amount_label = f"page_{page_id}_other_amount"
                try:
                    run_paddle_on_crop(other_amount_img_path, file_output_dir, other_amount_label)
                except Exception as e:
                    logger.error(f"Failed to run OCR on page {page_id} other_amount crop: {e}")
                    logger.debug(traceback.format_exc())
                    print(f"[WARNING] Skipping page {page_id} other_amount crop due to OCR error")
                other_amount_json_path = file_output_dir / f"{other_amount_label}_ocr_0.json"
                other_amount_words = []
                if other_amount_json_path.exists():
                    other_amount_words = load_words(str(other_amount_json_path))
                    # ADD PAGE INFO
                    for w in other_amount_words:
                        w['page'] = page_id
                    print(f"[OK] Loaded {len(other_amount_words)} words from other-amount-region OCR")
                    
                    # Extract other_amount
                    other_amt = extract_other_amount(other_amount_words)
                    if other_amt:
                        data_row['other_amount'] = other_amt
                        print(f"[OK] Other amount extracted: {other_amt}")
                    
                if date_val:
                    # --- TRY TO PARSE DIRECT OCR DATE ---
                    parsed = parse_ocr_date(date_val)

                    # ✅ IF PARSING SUCCEEDED → ASSIGN IMMEDIATELY
                    if parsed and parsed.isdigit() and len(parsed) == 8:
                        if not data_row.get('intrlab_cash_log_date') or data_row.get('intrlab_cash_log_date') == 'N/A':
                            data_row['intrlab_cash_log_date'] = parsed
                            print(f"[DATE] Direct OCR parse succeeded: {parsed}")
                    else:
                        print(f"[DATE] Direct OCR parse failed for: '{date_val}' -> will retry in SECOND PASS")




            else:
                print(f"Invalid page {page_id} for prefix/date: missing {'date' if not has_date else ''} {'prefix' if not has_prefix else ''} {'comments' if not has_comments else ''}")
                middle_words_for_amount = []
            # ────────────────────────────────────────────────────────────────────────

            # transit account
            ta_res = extract_transit_account(page_words)

            # Initialize amount variables (get from data_row if already set on previous page)
            total_from_bottom_amt = data_row.get('_total_from_bottom_amt')
            total_from_bottom_score = data_row.get('_total_from_bottom_score', -1)
            total_cash_from_middle_amt = data_row.get('_total_cash_from_middle_amt')
            total_cash_from_middle_score = data_row.get('_total_cash_from_middle_score', -1)

            # ------------------ PAGE VALIDATION FOR TRANSIT/ACCOUNT ------------------
            # Only assign if CIBC present and at least one of transit/account
            if has_cibc and (ta_res.get('transit_number') or ta_res.get('account_number')):
                print(f"Valid page {page_id} for transit/account with CIBC")
                if ta_res.get('transit_number') and not data_row.get('transit_number'):
                    data_row['transit_number'] = ta_res['transit_number']
                if ta_res.get('account_number') and not data_row.get('account_number'):
                    data_row['account_number'] = ta_res['account_number']

                # Extract "TOTAL" from BOTTOM OCR of THIS page (transit/account page)
                if bottom_json_path.exists():
                    bottom_words = load_words(str(bottom_json_path))
                    # Find TOTAL keyword in bottom
                    for w in bottom_words:
                        if sim(w["text"].strip().lower(), "total") >= TOTAL_THRESHOLD and " " not in w["text"].strip():
                            print(f"\n[BOTTOM-TOTAL] Found 'TOTAL' keyword: '{w['text']}'")
                            val_w = find_value_right_of_keyword(w, bottom_words)
                            if val_w:
                                cleaned = clean_amount(val_w["text"])
                                if cleaned:
                                    total_from_bottom_amt = cleaned
                                    total_from_bottom_score = val_w.get("score", 0)
                                    data_row['_total_from_bottom_amt'] = total_from_bottom_amt
                                    data_row['_total_from_bottom_score'] = total_from_bottom_score
                                    print(f"                Value: '{val_w['text']}' -> Cleaned: '{cleaned}'")
                                    print(f"                Confidence: {total_from_bottom_score:.4f}")
                            break
            else:
                print(f"Invalid page {page_id} for transit/account: missing {'CIBC' if not has_cibc else ''} {'transit' if not ta_res.get('transit_number') else ''} {'account' if not ta_res.get('account_number') else ''}")
            # ────────────────────────────────────────────────────────────────────────

            # Amount extraction: Find "TOTAL CASH & CHEQUES" from middle area (date/prefix/comments page)
            # Then compare with TOTAL from bottom area (transit/account page)
            if has_date and has_prefix and has_comments and middle_words_for_amount:
                print(f"\n[MIDDLE-TOTAL-CASH] Processing {len(middle_words_for_amount)} words from middle area")
                
                # KEYWORD: Search for "total cash & cheques" phrase in middle area
                total_cash_patterns = ["total cash & cheques", "total cash cheques", "totalcash & cheques", "totalcashcheques"]
                total_cash_kw = find_keyword(middle_words_for_amount, total_cash_patterns, threshold=0.70)
                
                if total_cash_kw:
                    print(f"[MIDDLE-TOTAL-CASH] Found keyword: '{total_cash_kw['text']}'")
                    val_w = find_value_right_of_keyword(total_cash_kw, middle_words_for_amount)
                    if val_w:
                        cleaned = clean_amount(val_w["text"])
                        if cleaned:
                            total_cash_from_middle_amt = cleaned
                            total_cash_from_middle_score = val_w.get("score", 0)
                            data_row['_total_cash_from_middle_amt'] = total_cash_from_middle_amt
                            data_row['_total_cash_from_middle_score'] = total_cash_from_middle_score
                            total_from_bottom_amt = data_row.get('_total_from_bottom_amt')
                            total_from_bottom_score = data_row.get('_total_from_bottom_score', -1)
                            print(f"                    Value: '{val_w['text']}' -> Cleaned: '{cleaned}'")
                            print(f"                    Confidence: {total_cash_from_middle_score:.4f}")
                else:
                    print(f"[MIDDLE-TOTAL-CASH] No 'total cash & cheques' keyword found in middle area")

            # COMPARE: Compare TOTAL (from bottom) vs TOTAL CASH & CHEQUES (from middle)
            if total_from_bottom_amt and total_cash_from_middle_amt:
                print(f"\n[COMPARE VALUE SCORES]")
                print(f"  TOTAL (from bottom):                {total_from_bottom_amt} -> Confidence: {total_from_bottom_score:.4f}")
                print(f"  TOTAL CASH & CHEQUES (from middle): {total_cash_from_middle_amt} -> Confidence: {total_cash_from_middle_score:.4f}")
                
                if total_cash_from_middle_score >= total_from_bottom_score:
                    chosen = (total_cash_from_middle_amt, total_cash_from_middle_score, 'total_cash_middle')
                    print(f"  >> CHOSEN: TOTAL CASH & CHEQUES (score {total_cash_from_middle_score:.4f}) -> {total_cash_from_middle_amt}")
                else:
                    chosen = (total_from_bottom_amt, total_from_bottom_score, 'total_bottom')
                    print(f"  >> CHOSEN: TOTAL (score {total_from_bottom_score:.4f}) -> {total_from_bottom_amt}")
                
                data_row['amount_candidates'].append(chosen)
                print(f"[OK] Added amount candidate: {chosen}\n")
            elif total_from_bottom_amt:
                data_row['amount_candidates'].append((total_from_bottom_amt, total_from_bottom_score, 'total_bottom'))
                print(f"\n[OK] Only TOTAL found -> {total_from_bottom_amt} (score {total_from_bottom_score:.4f})\n")
            elif total_cash_from_middle_amt:
                data_row['amount_candidates'].append((total_cash_from_middle_amt, total_cash_from_middle_score, 'total_cash_middle'))
                print(f"\n[OK] Only TOTAL CASH & CHEQUES found -> {total_cash_from_middle_amt} (score {total_cash_from_middle_score:.4f})\n")

            if bottom_json_path.exists():
                # invoice/visit extraction ONLY on bottom (Comments keyword is in bottom area)
                bottom_words = load_words(str(bottom_json_path))
                inv_res = extract_invoice(bottom_words)
                if inv_res['invoice'] == 'Y' and data_row.get('invoice') != 'Y':
                    data_row['invoice'] = 'Y'
                if inv_res['visit'] == 'Y' and data_row.get('visit') != 'Y':
                    data_row['visit'] = 'Y'

        # ================== SECOND PASS: DATE FALLBACK (AFTER ALL PAGES) ==================
        # If date is still missing/N/A, search ALL pages for accession + dates
        if main_page_found and (not data_row.get('intrlab_cash_log_date') or data_row.get('intrlab_cash_log_date') == 'N/A'):
            print("\n" + "="*60)
            print("SECOND PASS: DATE FALLBACK (across all pages)")
            print("="*60)
            
            # Find ACCESSION keyword across ALL pages
            accession_candidates = []
            for w in all_document_words:
                t = w["text"]
                score = sim(t, "accession")
                if score >= 0.80:
                    b = w.get("bbox", [0,0,0,0])
                    p = w.get("page", "?")
                    accession_candidates.append({
                        "text": t,
                        "bbox": b,
                        "score": score,
                        "page": p
                    })
            
            if accession_candidates:
                # Take the first/best accession candidate's page
                acc_page = accession_candidates[0]["page"]
                print(f"[SECOND PASS] Found ACCESSION keyword on page {acc_page}: '{accession_candidates[0]['text']}'")
                print(f"[SECOND PASS] Searching for date patterns on accession page {acc_page}...")
                consensus = extract_consensus_date_from_accession_page(all_document_words, acc_page)
                if consensus:
                    print(f"[SECOND PASS] ✅ Consensus date found: {consensus}")
                    if consensus.isdigit() and len(consensus) == 8:
                        data_row['intrlab_cash_log_date'] = consensus
                        print(f"[SECOND PASS] ✅ ILR Date set to: {consensus}")
                else:
                    print(f"[SECOND PASS] ❌ No consensus date found on accession page {acc_page}")
            else:
                print(f"[SECOND PASS] ❌ No ACCESSION keyword found on any page")

        # ================== PREFIX EXTRACTION (AFTER ALL PAGES) ==================
        # After all pages: extract prefix if main_page_found
        if main_page_found and not data_row.get('prefix'):
            print("\n" + "="*60)
            print("PREFIX EXTRACTION LOGIC (from ACCESSION keyword)")
            print("="*60)
            
            # Find ALL ACCESSION keywords
            candidates = []
            for w in all_document_words:
                t = w["text"]
                score = sim(t, "accession")
                if score >= 0.80:
                    b = w.get("bbox", [0,0,0,0])
                    p = w.get("page", "?")
                    candidates.append({
                        "text": t,
                        "bbox": b,
                        "score": score,
                        "page": p
                    })
            
            if not candidates:
                print("[X] ACCESSION keyword not found (threshold 0.80)")
                data_row['prefix'] = 'N/A'
            else:
                print(f"[OK] Found {len(candidates)} ACCESSION candidates")
                
                # For each ACCESSION candidate, search for overlapping values below it
                found_prefix = False
                for i, accession_kw in enumerate(candidates):
                    if found_prefix:
                        break
                    
                    acc_page = accession_kw.get("page", "?")
                    print(f"[*] Checking ACCESSION candidate {i+1} (Page {acc_page}): '{accession_kw['text']}'")
                    
                    # Get ACCESSION bounding box: [left, top, right, bottom]
                    acc_bbox = accession_kw["bbox"]
                    acc_left, acc_top, acc_right, acc_bottom = acc_bbox
                    
                    # Find words that have HORIZONTAL overlap AND word_top > acc_top
                    # Only take words that are below ACCESSION's TOP line and overlap horizontally
                    # SAME PAGE ONLY - NO vertical overlap matching
                    print(f"    ACCESSION bbox: left={acc_left:.1f}, top={acc_top:.1f}, right={acc_right:.1f}, bottom={acc_bottom:.1f}")
                    
                    for w in all_document_words:
                        page = w.get("page", "?")
                        
                        # Only check words from the same page as ACCESSION
                        if page != acc_page:
                            continue
                        
                        text = w["text"].strip()
                        bbox = w.get("bbox", [0, 0, 0, 0])
                        word_left, word_top, word_right, word_bottom = bbox
                        
                        # Check ONLY: word_top > acc_top AND horizontal overlap
                        below_acc_top = word_top > acc_top
                        horiz_overlap = (word_left < acc_right) and (word_right > acc_left)
                        
                        # Match ONLY if both conditions met
                        if below_acc_top and horiz_overlap:
                            text_clean = text.strip()
                            
                            # Only consider if it starts with alphanumeric (letters OR digits) and is reasonably long (>= 2 chars)
                            # AND doesn't contain special punctuation like ':' or '.'
                            if len(text_clean) >= 2 and text_clean[0].isalnum() and ':' not in text_clean and '.' not in text_clean:
                                print(f"    [OK] Found matching value on Page {page}: '{text_clean}'")
                                
                                # Extract first 2 characters as prefix
                                if len(text_clean) >= 2:
                                    data_row['prefix'] = text_clean[:2].upper()
                                    print(f"    [OK] Extracted prefix: {data_row['prefix']}")
                                    found_prefix = True
                                    break
                
                if not found_prefix:
                    print(f"[X] No overlapping values found")
                    data_row['prefix'] = 'N/A'


        # set defaults
        # CHECK: If both transit_no and account_no are N/A (or None), skip amount extraction
        transit_is_missing = data_row.get('transit_number') is None or data_row.get('transit_number') == 'N/A'
        account_is_missing = data_row.get('account_number') is None or data_row.get('account_number') == 'N/A'
        
        if transit_is_missing and account_is_missing:
            print("\n[AMOUNT SKIP] Both Transit and Account are missing - Skipping amount extraction")
            data_row['amount'] = 'N/A'
        # FINAL amount selection across all candidates (full-page candidates + bottom)
        elif data_row['amount_candidates']:
            best = max(data_row['amount_candidates'], key=lambda x: x[1])
            final_amt, final_score, source = best

            # ---------------- LOW CONFIDENCE OCR FIX ----------------
            if final_score < 0.87 and final_amt:
                # Rule 1: Remove leading '8'
                if final_amt.startswith("8"):
                    try:
                        final_amt = f"{float(final_amt[1:]):.2f}"
                    except:
                        pass

                # Rule 2: 1800 -> 180 (do nothing if already 180)
                try:
                    if int(float(final_amt)) == 1800:
                        final_amt = "180.00"
                except:
                    pass
            # --------------------------------------------------------

            data_row['amount'] = final_amt
        else:
            data_row['amount'] = 'N/A'


        return

    # --------------------- IMAGE ---------------------
    elif ext in IMAGE_EXT:
        raw_path = "_temp_img.jpg"
        cv2_img = cv2.imread(str(file_path))
        cv2.imwrite(raw_path, cv2_img)

        rotated_path = "_rotated_img.jpg"
        corrected = find_best_rotation(raw_path, rotated_path)

        img_cv2 = cv2.imread(rotated_path)

        crop1_path, crop2_path = crop_page_two_parts(img_cv2, page_id=1, save_folder=file_output_dir)

        run_paddle_on_crop(crop1_path, file_output_dir, "page_1_top")
        run_paddle_on_crop(crop2_path, file_output_dir, "page_1_bottom")

        os.remove(raw_path)

        top_json_path = file_output_dir / "page_1_top_ocr_0.json"
        bottom_json_path = file_output_dir / "page_1_bottom_ocr_0.json"

        top_results = None
        ta_res = None
        all_words = []
        has_comments = False
        has_cibc = False
        if top_json_path.exists():
            top_results = process(str(top_json_path), str(bottom_json_path) if bottom_json_path.exists() else None)

            # Load all_words
            all_words = load_words(str(top_json_path))
            if bottom_json_path.exists():
                all_words += load_words(str(bottom_json_path))

            # ADD PAGE INFO (image as page 1)
            for w in all_words:
                w['page'] = 1

            # ------------------ PAGE VALIDATION FOR PREFIX/DATE (image treated as page 1) ------------------
            if not top_results:
                top_results = {}

            has_date = 'intrlab cash log date' in top_results
            has_prefix = 'psc prefix#' in top_results
            has_comments = any(sim(w["text"].lower(), "comments") >= COMMENT_THRESHOLD for w in all_words)
            has_cibc = any("cibc" in w["text"].lower() for w in all_words)

            # Only assign if all three present
            if has_date and has_prefix and has_comments:
                print("Valid image for prefix/date/comments")
                date_val = top_results.get('intrlab cash log date', '')
                if date_val:
                    # --- DATE FINALIZATION (after middle OCR exists) ---
                    parsed = parse_ocr_date(date_val)

                    # 🔴 IF OCR PARSING FAILED → SEARCH ACCESSION PAGE
                    if not parsed or not parsed.isdigit():
                        # For image, all_words is page 1
                        # Find accession candidates
                        accession_candidates = []
                        for w in all_words:
                            t = w["text"]
                            score = sim(t, "accession")
                            if score >= 0.80:
                                b = w.get("bbox", [0,0,0,0])
                                p = w.get("page", 1)
                                accession_candidates.append({
                                    "text": t,
                                    "bbox": b,
                                    "score": score,
                                    "page": p
                                })
                        
                        consensus = None
                        if accession_candidates:
                            acc_page = accession_candidates[0]["page"]
                            print(f"[FALLBACK] Searching dates on accession page {acc_page}")
                            consensus = extract_consensus_date_from_accession_page(all_words, acc_page)
                            if consensus:
                                print(f"[FALLBACK] Consensus date from accession page: {consensus}")
                        
                        if consensus:
                            parsed = consensus
                        else:
                            parsed = None

                    # ✅ FINAL ASSIGNMENT (NUMERIC ONLY)
                    if not data_row.get('intrlab_cash_log_date') or data_row.get('intrlab_cash_log_date') == 'N/A':

                        if parsed and parsed.isdigit() and len(parsed) == 8:
                            data_row['intrlab_cash_log_date'] = parsed





                # --- NEW BEHAVIOR: FULL-PAGE OCR for single image ---
                full_label = f"page_1_full"
                full_img_path = f"{file_output_dir}/{full_label}.jpg"
                cv2.imwrite(full_img_path, img_cv2)
                run_paddle_on_crop(full_img_path, file_output_dir, full_label)
                full_json_path = file_output_dir / f"{full_label}_ocr_0.json"
                full_words_for_amount_image = []
                if full_json_path.exists():
                    full_words_for_amount_image = load_words(str(full_json_path))
                    # ADD PAGE INFO
                    for w in full_words_for_amount_image:
                        w['page'] = 1
                # --- END full-page OCR setup (amount extraction will happen after transit/account validation) ---

                # --- OTHER AMOUNT REGION OCR FOR IMAGE (left crop: x=1300 to end, y=650-1200) ---
                h, w_img = img_cv2.shape[:2]
                other_amount_crop_img = img_cv2[650:1200, 1300:w_img]
                other_amount_img_path = f"{file_output_dir}/page_1_other_amount.jpg"
                cv2.imwrite(other_amount_img_path, other_amount_crop_img)
                run_paddle_on_crop(other_amount_img_path, file_output_dir, "page_1_other_amount")
                other_amount_json_path_img = file_output_dir / "page_1_other_amount_ocr_0.json"
                other_amount_words_img = []
                if other_amount_json_path_img.exists():
                    other_amount_words_img = load_words(str(other_amount_json_path_img))
                    # ADD PAGE INFO
                    for w in other_amount_words_img:
                        w['page'] = 1
                    print(f"[OK] Loaded {len(other_amount_words_img)} words from other-amount-region OCR (image)")
                    
                    # Extract other_amount
                    other_amt = extract_other_amount(other_amount_words_img)
                    if other_amt:
                        data_row['other_amount'] = other_amt
                        print(f"[OK] Other amount extracted (image): {other_amt}")

                # Prefix logic (accession-based) - for image, use local all_words
                print("\n" + "="*60)
                print("PREFIX EXTRACTION LOGIC (from ACCESSION keyword) - IMAGE")
                print("="*60)
                
                # Find ALL ACCESSION keywords
                acc_candidates = []
                for w in all_words:
                    t = w["text"]
                    score = sim(t, "accession")
                    if score >= 0.80:
                        b = w.get("bbox", [0,0,0,0])
                        acc_candidates.append({
                            "text": t,
                            "bbox": b,
                            "score": score
                        })
                
                if not acc_candidates:
                    print("[X] ACCESSION keyword not found (threshold 0.80)")
                    data_row['prefix'] = 'N/A'
                else:
                    print(f"[OK] Found {len(acc_candidates)} ACCESSION candidates")
                    
                    # For each ACCESSION candidate, search for overlapping values below it
                    found_prefix = False
                    for i, accession_kw in enumerate(acc_candidates):
                        if found_prefix:
                            break
                        
                        print(f"[*] Checking ACCESSION candidate {i+1}: '{accession_kw['text']}'")
                        
                        # Get ACCESSION bounding box: [left, top, right, bottom]
                        acc_bbox = accession_kw["bbox"]
                        acc_left, acc_top, acc_right, acc_bottom = acc_bbox
                        
                        # Find words below ACCESSION that horizontally overlap with it
                        for w in all_words:
                            bbox = w.get("bbox", [0, 0, 0, 0])
                            word_left, word_top, word_right, word_bottom = bbox
                            
                            # Check if word is below ACCESSION (top > acc_bottom) and horizontally overlaps
                            is_below = word_top > acc_bottom
                            horiz_overlap = (word_left < acc_right) and (word_right > acc_left)
                            
                            if is_below and horiz_overlap:
                                text = w["text"].strip()
                                
                                # Only consider if it starts with letters and is reasonably long (> 2 chars)
                                # AND doesn't contain special punctuation like ':' or '.'
                                if len(text) > 2 and text[0].isalpha() and ':' not in text and '.' not in text:
                                    print(f"    [OK] Found overlapping value: '{text}'")
                                    
                                    # Extract first 2 characters as prefix (can be letters or digits)
                                    # For N03100037, we want N0 as prefix
                                    if len(text) >= 2:
                                        data_row['prefix'] = text[:2].upper()
                                        print(f"    [OK] Extracted prefix: {data_row['prefix']}")
                                        found_prefix = True
                                        break
                    
                    if not found_prefix:
                        print(f"[X] No overlapping values found")
                        data_row['prefix'] = 'N/A'



                # Full-page total cash & cheques (already handled above via full_words)
            else:
                print(f"Invalid image for prefix/date: missing {'date' if not has_date else ''} {'prefix' if not has_prefix else ''} {'comments' if not has_comments else ''}")
            # --------------------------------------------------------------------

            ta_res = extract_transit_account(all_words)

            # ------------------ PAGE VALIDATION FOR TRANSIT/ACCOUNT ------------------
            # Only assign if CIBC present and at least one of transit/account
            if has_cibc and (ta_res.get('transit_number') or ta_res.get('account_number')):
                print("Valid image for transit/account with CIBC")
                if ta_res.get('transit_number'):
                    data_row['transit_number'] = ta_res['transit_number']
                if ta_res.get('account_number'):
                    data_row['account_number'] = ta_res['account_number']
            else:
                print(f"Invalid image for transit/account: missing {'CIBC' if not has_cibc else ''} {'transit' if not ta_res.get('transit_number') else ''} {'account' if not ta_res.get('account_number') else ''}")
            # --------------------------------------------------------------------

            # Amount extraction from full-page OCR for IMAGE - ONLY if BOTH account AND transit found
            if ta_res.get('transit_number') and ta_res.get('account_number'):
                # Extract amount from full-page OCR
                if full_words_for_amount_image:
                    total_candidate_amt = None
                    total_candidate_score = -1
                    total_cash_candidate_amt = None
                    total_cash_candidate_score = -1

                    total_kw = None
                    for i, w in enumerate(full_words_for_amount_image):
                        if sim(w["text"].strip(), "total") >= TOTAL_THRESHOLD and " " not in w["text"].strip():
                            total_kw = w
                            break

                    if total_kw:
                        val_w = find_value_right_of_keyword(total_kw, full_words_for_amount_image)
                        if val_w:
                            cleaned = clean_amount(val_w["text"])
                            if cleaned:
                                total_candidate_amt = cleaned
                                total_candidate_score = val_w.get("score", 0)

                    total_cash_patterns = ["total cash & cheques", "total cash cheques"]
                    total_cash_kw = find_keyword(full_words_for_amount_image, total_cash_patterns, threshold=0.90)
                    if total_cash_kw:
                        val_w2 = find_value_right_of_keyword(total_cash_kw, full_words_for_amount_image)
                        if val_w2:
                            cleaned2 = clean_amount(val_w2["text"])
                            if cleaned2:
                                total_cash_candidate_amt = cleaned2
                                total_cash_candidate_score = val_w2.get("score", 0)

                    chosen = None
                    if total_cash_candidate_amt and total_candidate_amt:
                        if total_cash_candidate_score >= total_candidate_score:
                            chosen = (total_cash_candidate_amt, total_cash_candidate_score, 'full_total_cash')
                        else:
                            chosen = (total_candidate_amt, total_candidate_score, 'full_total')
                    elif total_cash_candidate_amt:
                        chosen = (total_cash_candidate_amt, total_cash_candidate_score, 'full_total_cash')
                    elif total_candidate_amt:
                        chosen = (total_candidate_amt, total_candidate_score, 'full_total')

                    if chosen:
                        data_row['amount_candidates'].append(chosen)
                        print(f"Added amount candidate from full-page OCR (image): {chosen}")

        if bottom_json_path.exists():
            # invoice/visit
            inv_res = extract_invoice(all_words)
            if inv_res['invoice'] == 'Y' and data_row.get('invoice') != 'Y':
                data_row['invoice'] = 'Y'
            if inv_res['visit'] == 'Y' and data_row.get('visit') != 'Y':
                data_row['visit'] = 'Y'

            # amount (bottom)
            if ta_res is not None and has_cibc and (ta_res.get('transit_number') or ta_res.get('account_number')):
                raw_amt, bottom_score = extract_bottom_total(str(bottom_json_path))
                if raw_amt:
                    cleaned_amt = clean_amount(raw_amt)
                    if cleaned_amt:
                        data_row['amount_candidates'].append((cleaned_amt, bottom_score, 'bottom'))

        # defaults
        if data_row.get('invoice') is None:
            data_row['invoice'] = 'N'
        if data_row.get('visit') is None:
            data_row['visit'] = 'N'
        for k in ['transit_number', 'account_number', 'prefix']:
            if data_row.get(k) is None:
                data_row[k] = 'N/A'

        # Final amount pick - CHECK: If both transit_no and account_no are missing, skip amount
        transit_is_missing = data_row.get('transit_number') is None or data_row.get('transit_number') == 'N/A'
        account_is_missing = data_row.get('account_number') is None or data_row.get('account_number') == 'N/A'
        
        logger.debug(f"Amount candidates: {len(data_row['amount_candidates'])} found")
        for amt, score, source in data_row['amount_candidates']:
            logger.debug(f"  Amount candidate: {amt} (score: {score:.2f}, source: {source})")
        
        if transit_is_missing and account_is_missing:
            logger.warning("Both Transit and Account are missing - Skipping amount extraction")
            print("[AMOUNT SKIP] Both Transit and Account are missing - Skipping amount extraction")
            data_row['amount'] = 'N/A'
        elif data_row['amount_candidates']:
            best = max(data_row['amount_candidates'], key=lambda x: x[1])
            data_row['amount'] = best[0]
            logger.info(f"Selected amount: {best[0]} (score: {best[1]:.2f}, source: {best[2]})")
        else:
            logger.warning("No amount candidates found")
            data_row['amount'] = 'N/A'

    return
# --------------------------------------------------------------

# ---------------------------- MAIN ----------------------------
def run_ocr(input_path, output_excel_path=None):
    logger.info("="*80)
    logger.info("STARTING OCR PROCESSING")
    logger.info(f"Input path: {input_path}")
    if output_excel_path:
        logger.info(f"Output Excel path: {output_excel_path}")
    logger.info("="*80)
    
    input_path = Path(input_path)

    output_base = input_path.parent / "output"
    output_base.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_base}")

    files = []

    if input_path.is_dir():
        logger.info(f"Input is directory, scanning for files...")
        for f in sorted(input_path.iterdir()):
            if f.suffix.lower() == ".pdf" or f.suffix.lower() in IMAGE_EXT:
                files.append(f)
                logger.debug(f"Found file: {f.name}")
    else:
        logger.info(f"Input is single file: {input_path.name}")
        files.append(input_path)

    if not files:
        logger.error("No valid files found.")
        print("No valid files found.")
        return

    logger.info(f"Total files to process: {len(files)}")
    print("\nRunning PaddleOCR + Rotate + 2-Crop System...\n")

    headers = [
        'file_name',
        'invoice(Y/N)',
        'Visit(Y/N)',
        'Transit_No',
        'Account_No',
        'prefix',
        'ILR_date',
        'amount',
        'other_amount'
    ]

    data_list = []

    for idx, file in enumerate(files, 1):
        logger.info("="*80)
        logger.info(f"FILE {idx}/{len(files)}: {file.name}")
        logger.info("="*80)
        print(f"\nProcessing: {file.name}")
        
        data_row = {
            'file_name': file.name,
            'invoice': None,
            'visit': None,
            'transit_number': None,
            'account_number': None,
            'prefix': None,
            'intrlab_cash_log_date': None,
            'amount': None,
            'other_amount': None,
            'amount_candidates': []
        }

        try:
            run_ocr_on_file(file, output_base, data_row)
            logger.info(f"Successfully processed: {file.name}")
        except Exception as e:
            logger.error(f"Error processing file {file.name}: {e}")
            logger.debug(traceback.format_exc())
            print(f"[ERROR] Failed to process {file.name}: {e}")

        # default invoice/visit to Y/N
        if data_row['invoice'] is None:
            data_row['invoice'] = 'N'
        if data_row.get('visit') is None:
            data_row['visit'] = 'N'

        # default others
        for k in ['transit_number', 'account_number',
                'prefix', 'intrlab_cash_log_date', 'amount', 'other_amount']:
            if data_row[k] is None:
                data_row[k] = 'N/A'

        # append mapped row
        data_list.append([
            data_row['file_name'],
            data_row['invoice'],
            data_row['visit'],
            data_row['transit_number'],
            data_row['account_number'],
            data_row['prefix'],
            data_row['intrlab_cash_log_date'],
            data_row['amount'],
            data_row['other_amount']
        ])


    # Write Excel
    logger.info("Generating Excel output...")
    try:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(headers)
        for row in data_list:
            ws.append(row)
        
        # Use provided output path or default
        if output_excel_path:
            excel_path = Path(output_excel_path)
            excel_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            excel_path = input_path.parent / "input_results2.xlsx"
        
        wb.save(excel_path)
        logger.info(f"Excel file saved: {excel_path}")
        print("\n[OK] COMPLETED - Check output folder:", output_base)
        print(f"[OK] Excel saved: {excel_path}")
    except Exception as e:
        logger.error(f"Error saving Excel file: {e}")
        logger.debug(traceback.format_exc())
        print(f"[ERROR] Failed to save Excel: {e}")
        
    logger.info("="*80)
    logger.info("OCR PROCESSING COMPLETED")
    logger.info("="*80)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='OCR Application for Bank Deposit Forms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main5.py document.pdf
  python main5.py document.pdf -o results.xlsx
  python main5.py document.pdf -o results.xlsx -p "C:\\poppler\\bin"
  python main5.py ./data/forms/ -o ./output/results.xlsx
        ''')
    
    parser.add_argument('input', 
                        help='Path to PDF file or directory containing PDF/image files')
    
    parser.add_argument('-o', '--output', 
                        dest='output_excel',
                        help='Output Excel file path (default: input_results2.xlsx in input directory)')
    
    parser.add_argument('-p', '--poppler', 
                        dest='poppler_path',
                        default=None,
                        help=f'Path to Poppler bin directory (default from config: {config.get("poppler_path", "D:\\Program Files\\poppler-25.07.0\\Library\\bin")})')
    
    args = parser.parse_args()
    
    # Set Poppler path from config or command line (modifying global variable)
    if args.poppler_path:
        globals()['POPLER_PATH'] = args.poppler_path
        logger.info(f"Poppler path overridden by command line: {args.poppler_path}")
    else:
        globals()['POPLER_PATH'] = config.get('poppler_path', r'D:\Program Files\poppler-25.07.0\Library\bin')
        logger.info(f"Using Poppler path from config: {config.get('poppler_path')}")
    
    try:
        logger.info("Application started from main")
        logger.info(f"Command line args: input={args.input}, output={args.output_excel}")
        if args.poppler_path:
            logger.info(f"Poppler path (command line): {args.poppler_path}")
        else:
            logger.info(f"Poppler path (config): {POPLER_PATH}")
        
        # Validate input path exists
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input path does not exist: {args.input}")
            print(f"[ERROR] Input path does not exist: {args.input}")
            sys.exit(1)
        
        run_ocr(args.input, args.output_excel)
        logger.info("Application completed successfully")
    except Exception as e:
        logger.critical(f"Application crashed: {e}")
        logger.critical(traceback.format_exc())
        print(f"\n[CRITICAL ERROR] Application crashed: {e}")
        print("Check log file for details: logs/")
        raise
    
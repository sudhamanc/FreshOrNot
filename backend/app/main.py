import io
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

try:
    import torch
    import torchvision.transforms as T
except ImportError as exc:
    raise RuntimeError('PyTorch and torchvision are required for freshness inference.') from exc

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise RuntimeError('ultralytics is required for stage-1 detector inference.') from exc

APP_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DETECTOR_MODEL = APP_ROOT / 'model' / 'detector' / 'stage1_yolov8_best.pt'
DEFAULT_DETECTOR_CLASSES = APP_ROOT / 'model' / 'detector' / 'stage1_yolo_classes.json'

DEFAULT_FRESHNESS_MODEL = APP_ROOT / 'model' / 'freshness_binary.pt'
DEFAULT_FRESHNESS_CLASSES = APP_ROOT / 'model' / 'freshness_binary_classes.json'

DETECTOR_MODEL_PATH = Path(os.getenv('DETECTOR_MODEL_PATH', str(DEFAULT_DETECTOR_MODEL)))
DETECTOR_CLASSES_PATH = Path(os.getenv('DETECTOR_CLASSES_PATH', str(DEFAULT_DETECTOR_CLASSES)))

FRESHNESS_MODEL_PATH = Path(os.getenv('FRESHNESS_MODEL_PATH', str(DEFAULT_FRESHNESS_MODEL)))
FRESHNESS_CLASSES_PATH = Path(os.getenv('FRESHNESS_CLASSES_PATH', str(DEFAULT_FRESHNESS_CLASSES)))

DETECTOR_CONF_THRESHOLD = float(os.getenv('DETECTOR_CONF_THRESHOLD', '0.35'))
DETECTOR_NMS_THRESHOLD = float(os.getenv('DETECTOR_NMS_THRESHOLD', '0.45'))
FRESHNESS_UNKNOWN_THRESHOLD = float(os.getenv('FRESHNESS_UNKNOWN_THRESHOLD', '0.60'))

PRODUCE_PROFILES: dict[str, dict[str, int]] = {
    'apple': {'fresh_max': 10, 'stale_threshold': 4},
    'banana': {'fresh_max': 7, 'stale_threshold': 3},
    'tomato': {'fresh_max': 8, 'stale_threshold': 3},
    'strawberry': {'fresh_max': 5, 'stale_threshold': 2},
    'broccoli': {'fresh_max': 7, 'stale_threshold': 3},
    'carrot': {'fresh_max': 14, 'stale_threshold': 5},
    'mango': {'fresh_max': 6, 'stale_threshold': 2},
    'orange': {'fresh_max': 14, 'stale_threshold': 5},
    'pepper': {'fresh_max': 10, 'stale_threshold': 4},
    'bittergourd': {'fresh_max': 5, 'stale_threshold': 2},
    'capsicum': {'fresh_max': 10, 'stale_threshold': 4},
    'cucumber': {'fresh_max': 7, 'stale_threshold': 3},
    'okra': {'fresh_max': 5, 'stale_threshold': 2},
    'potato': {'fresh_max': 21, 'stale_threshold': 7},
}

PRODUCE_ALIASES = {
    'apple': 'apple',
    'apples': 'apple',
    'banana': 'banana',
    'bananas': 'banana',
    'orange': 'orange',
    'oranges': 'orange',
    'tomato': 'tomato',
    'tomatoes': 'tomato',
    'potato': 'potato',
    'potatoes': 'potato',
    'cucumber': 'cucumber',
    'cucumbers': 'cucumber',
    'okra': 'okra',
    'bittergroud': 'bittergourd',
    'bittergourd': 'bittergourd',
    'capsicum': 'capsicum',
    'capciscum': 'capsicum',
    'bellpepper': 'pepper',
    'bellpeppers': 'pepper',
    'pepper': 'pepper',
    'peppers': 'pepper',
    'carrot': 'carrot',
    'carrots': 'carrot',
    'mango': 'mango',
    'mangoes': 'mango',
    'strawberry': 'strawberry',
    'strawberries': 'strawberry',
}

FRESHNESS_LABELS = ['fresh', 'stale']

PT_TRANSFORM = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def parse_origins() -> list[str]:
    origins = os.getenv('CORS_ORIGINS', '*').strip()
    if origins == '*':
        return ['*']
    return [item.strip() for item in origins.split(',') if item.strip()]


app = FastAPI(title='FreshOrNot Inference API', version='3.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_origins(),
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)

DETECTOR_MODEL: Any | None = None
DETECTOR_CLASSES: list[str] = []
DETECTOR_ERROR: str | None = None

FRESHNESS_MODEL: Any | None = None
FRESHNESS_ERROR: str | None = None


@dataclass
class Detection:
    produce: str
    confidence: float
    bbox: tuple[int, int, int, int]


def _normalize_produce_name(raw_name: str) -> str:
    token = re.sub(r'[^a-zA-Z0-9]+', '', raw_name).lower()
    return PRODUCE_ALIASES.get(token, token)


def _load_detector() -> None:
    global DETECTOR_MODEL, DETECTOR_CLASSES, DETECTOR_ERROR
    if not DETECTOR_MODEL_PATH.exists():
        DETECTOR_MODEL = None
        DETECTOR_ERROR = f'Detector model missing: {DETECTOR_MODEL_PATH}'
        return

    try:
        model = YOLO(str(DETECTOR_MODEL_PATH))
        DETECTOR_MODEL = model
        if DETECTOR_CLASSES_PATH.exists():
            payload = json.loads(DETECTOR_CLASSES_PATH.read_text(encoding='utf-8'))
            if isinstance(payload, list) and all(isinstance(item, str) for item in payload):
                DETECTOR_CLASSES = payload
            else:
                DETECTOR_CLASSES = []
        else:
            names = getattr(model, 'names', {})
            if isinstance(names, dict):
                DETECTOR_CLASSES = [str(names[idx]) for idx in sorted(names)]
            elif isinstance(names, list):
                DETECTOR_CLASSES = [str(item) for item in names]
            else:
                DETECTOR_CLASSES = []
        DETECTOR_ERROR = None
    except Exception as exc:  # noqa: BLE001
        DETECTOR_MODEL = None
        DETECTOR_CLASSES = []
        DETECTOR_ERROR = f'Failed to load detector: {exc}'


def _load_freshness() -> None:
    global FRESHNESS_MODEL, FRESHNESS_ERROR
    if not FRESHNESS_MODEL_PATH.exists():
        FRESHNESS_MODEL = None
        FRESHNESS_ERROR = f'Freshness model missing: {FRESHNESS_MODEL_PATH}'
        return

    try:
        model = torch.load(FRESHNESS_MODEL_PATH, map_location='cpu', weights_only=False)
        model.eval()
        FRESHNESS_MODEL = model

        if FRESHNESS_CLASSES_PATH.exists():
            payload = json.loads(FRESHNESS_CLASSES_PATH.read_text(encoding='utf-8'))
            if isinstance(payload, list) and len(payload) == 2 and all(isinstance(item, str) for item in payload):
                global FRESHNESS_LABELS
                FRESHNESS_LABELS = [payload[0].lower(), payload[1].lower()]

        FRESHNESS_ERROR = None
    except Exception as exc:  # noqa: BLE001
        FRESHNESS_MODEL = None
        FRESHNESS_ERROR = f'Failed to load freshness model: {exc}'


@app.on_event('startup')
def startup() -> None:
    _load_detector()
    _load_freshness()


@app.get('/api/health')
def health() -> dict[str, Any]:
    return {
        'status': 'ok',
        'detector_ready': DETECTOR_MODEL is not None,
        'detector_error': DETECTOR_ERROR,
        'freshness_ready': FRESHNESS_MODEL is not None,
        'freshness_error': FRESHNESS_ERROR,
        'detector_model_path': str(DETECTOR_MODEL_PATH),
        'detector_classes_path': str(DETECTOR_CLASSES_PATH),
        'freshness_model_path': str(FRESHNESS_MODEL_PATH),
    }


def _detect_produce(pil_img: Image.Image) -> Detection | None:
    if DETECTOR_MODEL is None:
        return None

    image_np = np.array(pil_img)
    results = cast(Any, DETECTOR_MODEL).predict(
        source=image_np,
        conf=DETECTOR_CONF_THRESHOLD,
        iou=DETECTOR_NMS_THRESHOLD,
        verbose=False,
    )
    if not results:
        return None

    result = results[0]
    boxes = getattr(result, 'boxes', None)
    if boxes is None or len(boxes) == 0:
        return None

    confs = boxes.conf.cpu().numpy()
    best = int(np.argmax(confs))

    xyxy = boxes.xyxy[best].cpu().numpy()
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x = max(0, x1)
    y = max(0, y1)
    bw = max(1, x2 - x)
    bh = max(1, y2 - y)

    class_idx = int(boxes.cls[best].item())
    raw_label = DETECTOR_CLASSES[class_idx] if class_idx < len(DETECTOR_CLASSES) else ''
    produce = _normalize_produce_name(raw_label)
    if not produce:
        produce = 'unknown'

    return Detection(produce=produce, confidence=float(confs[best]), bbox=(x, y, bw, bh))


def _predict_freshness(crop: Image.Image) -> tuple[str, float]:
    if FRESHNESS_MODEL is None:
        raise HTTPException(status_code=503, detail=f'Freshness model unavailable: {FRESHNESS_ERROR}')

    tensor = cast(torch.Tensor, PT_TRANSFORM(crop)).unsqueeze(0)
    with torch.no_grad():
        logits = FRESHNESS_MODEL(tensor)

    if isinstance(logits, (list, tuple)):
        logits = logits[0]

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    probs = torch.softmax(logits, dim=1).numpy()
    idx = int(np.argmax(probs[0]))
    confidence = float(probs[0][idx])
    label = FRESHNESS_LABELS[idx] if idx < len(FRESHNESS_LABELS) else 'fresh'

    return label, confidence


def _shelf_days(is_fresh: bool, score: float, profile: dict[str, int]) -> int:
    if is_fresh:
        return max(0, round(profile['stale_threshold'] + (profile['fresh_max'] - profile['stale_threshold']) * score))
    return max(0, round(profile['stale_threshold'] * (score / 0.42)))


@app.post('/api/predict')
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if DETECTOR_MODEL is None:
        raise HTTPException(status_code=503, detail=f'Detector unavailable: {DETECTOR_ERROR}')
    if FRESHNESS_MODEL is None:
        raise HTTPException(status_code=503, detail=f'Freshness model unavailable: {FRESHNESS_ERROR}')

    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='Upload a valid image file.')

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail='Received empty file.')

    try:
        image = Image.open(io.BytesIO(payload)).convert('RGB')
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f'Invalid image: {exc}') from exc

    detection = _detect_produce(image)
    if detection is None:
        return {
            'label': 'FRESH',
            'confidence': 0.0,
            'fresh_score': 0.0,
            'shelf_days': 0,
            'produce': 'unknown produce',
            'is_unknown': True,
            'unknown_reason': 'no_detection',
            'source': 'Two-stage offline: YOLO (.pt) + freshness classifier',
        }

    x, y, bw, bh = detection.bbox
    crop = image.crop((x, y, x + bw, y + bh))
    freshness_label, freshness_conf = _predict_freshness(crop)

    is_fresh = freshness_label == 'fresh'
    output_label = 'FRESH' if is_fresh else 'STALE'
    fresh_score = freshness_conf if is_fresh else 1.0 - freshness_conf

    is_unknown = detection.confidence < DETECTOR_CONF_THRESHOLD or freshness_conf < FRESHNESS_UNKNOWN_THRESHOLD
    profile = PRODUCE_PROFILES.get(detection.produce, {'fresh_max': 8, 'stale_threshold': 3})

    return {
        'label': output_label,
        'confidence': freshness_conf,
        'fresh_score': fresh_score,
        'shelf_days': _shelf_days(is_fresh, fresh_score, profile),
        'produce': 'unknown produce' if is_unknown else detection.produce,
        'is_unknown': is_unknown,
        'unknown_reason': 'low_confidence' if is_unknown else None,
        'source': 'Two-stage offline: YOLO (.pt) + freshness classifier',
        'pipeline': {
            'stage1': {
                'model': DETECTOR_MODEL_PATH.name,
                'produce': detection.produce,
                'confidence': detection.confidence,
                'bbox': [x, y, bw, bh],
            },
            'stage2': {
                'model': 'freshness_binary.pt',
                'label': output_label,
                'confidence': freshness_conf,
            },
        },
    }


@app.post('/api/feedback')
async def feedback(
    file: UploadFile = File(...),
    produce_name: str = Form(...),
    freshness_label: str = Form(...),
    notes: str = Form(''),
    predicted_label: str = Form(''),
    predicted_confidence: str = Form(''),
    is_unknown: str = Form('true'),
) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='Upload a valid image file.')

    normalized_freshness = freshness_label.strip().lower()
    if normalized_freshness not in {'fresh', 'stale'}:
        raise HTTPException(status_code=400, detail='freshness_label must be fresh or stale')

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail='Received empty file.')

    feedback_root = APP_ROOT / 'data' / 'feedback'
    feedback_raw = feedback_root / 'raw'
    feedback_jsonl = feedback_root / 'labels.jsonl'
    feedback_raw.mkdir(parents=True, exist_ok=True)

    try:
        image = Image.open(io.BytesIO(payload)).convert('RGB')
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f'Invalid image: {exc}') from exc

    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    image_id = f'{ts}_{uuid.uuid4().hex[:10]}'
    image_name = f'{image_id}.jpg'
    image_path = feedback_raw / image_name
    image.save(image_path, format='JPEG', quality=95)

    row = {
        'id': image_id,
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'image_path': str(image_path.relative_to(APP_ROOT)),
        'produce_name': produce_name,
        'freshness_label': normalized_freshness,
        'notes': notes.strip(),
        'predicted_label': predicted_label.strip(),
        'predicted_confidence': predicted_confidence.strip(),
        'is_unknown': is_unknown.strip().lower() == 'true',
    }

    with feedback_jsonl.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=True) + '\n')

    return {
        'status': 'stored',
        'id': image_id,
        'image_path': str(image_path.relative_to(APP_ROOT)),
    }

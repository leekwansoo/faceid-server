
import os, datetime, glob, pickle, cv2, uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi import UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from usb_device import list_usb_devices
from main import get_usb_camera, nvr_handler
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from ultralytics import YOLO
import numpy as np
# --- Face Embedding 라이브러리 선택 1: DeepFace (설치 쉬움) ---
from deepface import DeepFace

from main import _mjpeg_frames, _webcam_mjpeg, play_video, webcam_stream, get_usb_camera_index, get_usb_camera
camera = get_usb_camera()

from main import STATE, STATIC_DIR, BASE_DIR, CAMERA_RTSP, get_usb_camera_index

BASE_DIR = BASE_DIR
UPLOAD_DIR = BASE_DIR / "uploads"
STATIC_DIR = STATIC_DIR
STATE = STATE
MODEL_NAME = "Facenet512" # 가장 정확도 좋음, 또는 "VGG-Face", "ArcFace"
SAVE_DIR = "./captures"
LIBRARY_DIR = "./library" # label별 이미지 폴더
EMB_DB_PATH = "./face_db.pkl"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LIBRARY_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 얼굴용 YOLO 모델 로드 (처음 한번만)
yolo_face = YOLO("yolov8n-face.pt") 

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/config", StaticFiles(directory=str(BASE_DIR / "config")), name="config")
# Open your USB webcam
camera_index = get_usb_camera_index()
camera = cv2.VideoCapture(camera_index)


# 메모리에 로드될 DB: {label: [embedding_vector, ...]}
face_db = {}

def build_face_db():
    global face_db
    if os.path.exists(EMB_DB_PATH):
        with open(EMB_DB_PATH, "rb") as f:
            face_db = pickle.load(f)
        print(f"DB loaded: {list(face_db.keys())}")
        return

    face_db = {}
    for label in os.listdir(LIBRARY_DIR):
        label_path = os.path.join(LIBRARY_DIR, label)
        if not os.path.isdir(label_path): continue
        embs = []
        for img_path in glob.glob(os.path.join(label_path, "*.jpg")) + glob.glob(os.path.join(label_path, "*.png")):
            try:
                # DeepFace는 얼굴이 없으면 에러 -> enforce_detection=False
                emb = DeepFace.represent(img_path, model_name=MODEL_NAME, enforce_detection=False)[0]["embedding"]
                embs.append(np.array(emb))
            except Exception as e:
                print(f"skip {img_path}: {e}")
        if embs:
            face_db[label] = embs
    with open(EMB_DB_PATH, "wb") as f:
        pickle.dump(face_db, f)
    print(f"DB built: {face_db.keys()}")

def cosine_sim(a,b):
    return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9)

@app.on_event("startup")
def startup():
    build_face_db()

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "face-detection.html")

@app.get("/api/webcam/stream")
async def api_webcam_stream(webcam: int = 0):
    """[UPLINK] USB Webcam -> MJPEG stream for browser <img>"""
    if not camera.isOpened():
        return JSONResponse({"error": "Webcam not available"}, status_code=404)
    return StreamingResponse(_webcam_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/api/webcam/capture")
async def capture_and_search(file: UploadFile = File(...)):
    # 1. 원본 임시 저장
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVE_DIR, f"{timestamp}.jpg")
    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)

    # 2. YOLO Face detect
    from ultralytics import YOLO
    yolo_face = YOLO("yolov8n-face.pt") # 또는 당신의 모델
    results = yolo_face(path, conf=0.5, verbose=False)
    if len(results[0].boxes) == 0:
        return {"saved_path": path, "match": None, "msg": "no face"}

    with Image.open(path) as im:
        W, H = im.size
    x1,y1,x2,y2 = map(int, results[0].boxes[0].xyxy[0])

    # 3. enlarged box {x1-5, y1-5, x2+5, y2+5}
    x1_e = max(0, x1 - 40)
    y1_e = max(0, y1 - 40)
    x2_e = min(W, x2 + 40)
    y2_e = min(H, y2 + 40)
    box_size = {"x1": x1_e, "y1": y1_e, "x2": x2_e, "y2": y2_e}

    # 4. crop -> enlarged 이미지로 덮어쓰기 (요청하신 부분)
    with Image.open(path) as im:
        cropped = im.crop((x1_e, y1_e, x2_e, y2_e))
        # 인식률 위해 320x320 으로 리사이즈 권장
        cropped = cropped.resize((320, 320), Image.LANCZOS)
        cropped.save(path, "JPEG", quality=95)

    # 5. cropped 이미지로 library 검색
    try:
        query_emb = DeepFace.represent(path, model_name=MODEL_NAME, enforce_detection=False)[0]["embedding"]
        query_emb = np.array(query_emb)
    except Exception as e:
        return {"saved_path": path, "box": box_size, "error": str(e)}

    best_label = None
    best_score = -1
    all_scores = {}

    for label, emb_list in face_db.items():
        # label 내에서 가장 높은 점수
        scores = [cosine_sim(query_emb, db_emb) for db_emb in emb_list]
        max_score = max(scores) if scores else 0
        all_scores[label] = float(max_score)
        if max_score > best_score:
            best_score = max_score
            best_label = label

    # threshold - 0.4 이하면 unknown 처리
    THRESHOLD = 0.4
    if best_score < THRESHOLD:
        best_label = "unknown"

    return {
        "saved_path": path,
        "box": box_size,
        "match_label": best_label,}
    if not files:
        return {"error": "No files found for the given query"}

    return {"files": files}
async def get_captured_img(q: str):
    # search for the picture list with the given label name
    files = [file for file in os.listdir(SAVE_DIR) if file.startswith(q) and file.endswith(".jpg")]
    
    if not files:
        return {"error": "No files found for the given label"}

    return {"files": files}

@app.post("/api/library/rebuild")
def rebuild_db():
    if os.path.exists(EMB_DB_PATH):
        os.remove(EMB_DB_PATH)
    build_face_db()
    return {"status": "ok", "labels": list(face_db.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("face-recognition-server:app", host="127.0.0.1", port=8000, reload=True)
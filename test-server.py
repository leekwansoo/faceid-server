# Simple Test Server program for USB Camera Server


from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi import UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from usb_device import list_usb_devices
import cv2

# main.camera is the single capture handle used by _webcam_mjpeg
from main import camera, _webcam_mjpeg, get_usb_camera_index
from main import STATIC_DIR, BASE_DIR
BASE_DIR = BASE_DIR
STATIC_DIR = STATIC_DIR

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(request, "test.html")

@app.get("/webcam_index")
def get_webcam_index():
    camera_index = get_usb_camera_index()
    return {"camera_index": camera_index}

@app.get("/api/webcam/stream")
async def api_webcam_stream(webcam: int = 0):
    """[UPLINK] USB Webcam -> MJPEG stream for browser <img>"""
    if not camera.isOpened():
        return JSONResponse({"error": "Webcam not available"}, status_code=404)
    return StreamingResponse(_webcam_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("test-server:app", host="127.0.0.1", port=8000, reload=True)
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import json
from pathlib import Path

app = FastAPI(title="Real-Time Object Detection API")

model = YOLO("yolov8n.pt")

@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = Path("templates/index.html").read_text()
    return HTMLResponse(content=html_content)

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image file"}

    results = model(img)
    detections = []
    annotated = img.copy()

    for box in results[0].boxes:
        cls_id = int(box.cls)
        label = model.names[cls_id]
        conf = float(box.conf)
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        detections.append({
            "class": label,
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2]
        })

        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{label} {conf:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    _, buffer = cv2.imencode(".jpg", annotated)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "detections": detections,
        "count": len(detections),
        "annotated_image": f"data:image/jpeg;base64,{img_base64}"
    }

@app.websocket("/detect/stream")
async def detect_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            frame_b64 = payload.get("frame", "")

            img_data = base64.b64decode(frame_b64.split(",")[-1])
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                continue

            results = model(img, verbose=False)
            detections = []
            annotated = img.copy()

            for box in results[0].boxes:
                cls_id = int(box.cls)
                label = model.names[cls_id]
                conf = float(box.conf)
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                detections.append({
                    "class": label,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2]
                })

                color = (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{label} {conf:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            result_b64 = base64.b64encode(buffer).decode("utf-8")

            await websocket.send_text(json.dumps({
                "detections": detections,
                "count": len(detections),
                "annotated_frame": f"data:image/jpeg;base64,{result_b64}"
            }))

    except WebSocketDisconnect:
        print("WebSocket closed")

@app.get("/health")
async def health():
    return {"status": "ok", "model": "YOLOv8n"}
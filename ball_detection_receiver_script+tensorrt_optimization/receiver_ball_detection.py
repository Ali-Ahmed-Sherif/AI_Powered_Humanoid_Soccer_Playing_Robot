import cv2
import time
import threading
import queue
import requests
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

STREAM_URL = "http://10.172.251.96:81/stream"
OUTPUT_URL = "http://10.172.251.96:83/output"

MODEL_PATH = "best_26_s.engine"

CONF_THRESHOLD = 0.25
IMG_SIZE = 640

SEND_OUTPUT = True   # set False to test delay without HTTP output


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

model = YOLO(MODEL_PATH, task="detect")


# ─────────────────────────────────────────────
# LATEST FRAME READER
# ─────────────────────────────────────────────

class LatestFrameReader:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True

        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                time.sleep(0.01)
                continue

            with self.lock:
                self.latest_frame = frame

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None

            return True, self.latest_frame.copy()

    def release(self):
        self.running = False
        time.sleep(0.1)
        self.cap.release()


# ─────────────────────────────────────────────
# ASYNC OUTPUT SENDER
# ─────────────────────────────────────────────

output_queue = queue.Queue(maxsize=1)


def output_sender_loop():
    session = requests.Session()

    while True:
        payload = output_queue.get()

        if payload is None:
            break

        try:
            session.post(OUTPUT_URL, json=payload, timeout=0.03)
        except:
            pass


def send_output_async(payload):
    if not SEND_OUTPUT:
        return

    try:
        # Keep only the newest payload
        if output_queue.full():
            try:
                output_queue.get_nowait()
            except:
                pass

        output_queue.put_nowait(payload)

    except:
        pass


# ─────────────────────────────────────────────
# DRAW YOLO BOXES
# ─────────────────────────────────────────────

def draw_boxes(frame, result):
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        detections.append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": cx,
            "cy": cy
        })

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{class_name} {confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    return frame, detections


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    threading.Thread(target=output_sender_loop, daemon=True).start()

    reader = LatestFrameReader(STREAM_URL)

    print("Opening ESP32-CAM stream...")

    time.sleep(1)

    prev_time = time.time()

    while True:
        ret, frame = reader.read()

        if not ret:
            print("Waiting for camera frame...")
            time.sleep(0.05)
            continue

        frame = cv2.resize(frame, (640, 480))

        # YOLO inference on the latest available frame only
        results = model.predict(
            frame,
            conf=CONF_THRESHOLD,
            imgsz=IMG_SIZE,
            verbose=False
        )

        processed_frame, detections = draw_boxes(frame, results[0])

        if len(detections) > 0:
            best_detection = max(detections, key=lambda d: d["confidence"])

            payload = {
                "class_id": int(best_detection["class_id"]),
                "class_name": best_detection["class_name"],
                "confidence": float(best_detection["confidence"]),
                "cx": int(best_detection["cx"]),
                "cy": int(best_detection["cy"]),
                "x1": int(best_detection["x1"]),
                "y1": int(best_detection["y1"]),
                "x2": int(best_detection["x2"]),
                "y2": int(best_detection["y2"])
            }

            send_output_async(payload)

        # FPS display
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        cv2.putText(
            processed_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("ESP32-CAM YOLO Detection - Latest Frame", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    reader.release()
    output_queue.put(None)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
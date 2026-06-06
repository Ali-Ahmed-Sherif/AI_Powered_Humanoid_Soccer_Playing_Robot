import cv2
import time
import json
import argparse
from collections import deque
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stream",
        type=str,
        default="http://10.172.251.96:81/stream",
        help="ESP32-CAM MJPEG stream URL"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="best_26_s.engine",
        help="YOLO model path"
    )

    parser.add_argument(
        "--real-diameter",
        type=float,
        required=True,
        default=4.0,
        help="Real ball diameter in cm"
    )

    parser.add_argument(
        "--known-distance",
        type=float,
        required=True,
        help="Known distance from camera to ball in cm"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=30,
        help="Number of stable detections to average"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Frame resize width"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Frame resize height"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="ball_calibration.json",
        help="Output calibration JSON file"
    )

    return parser.parse_args()


def get_best_detection(result, model):
    if result.boxes is None or len(result.boxes) == 0:
        return None

    best_box = None
    best_conf = -1

    for box in result.boxes:
        confidence = float(box.conf[0])

        if confidence > best_conf:
            best_conf = confidence
            best_box = box

    if best_box is None:
        return None

    class_id = int(best_box.cls[0])
    class_name = model.names[class_id]

    x1, y1, x2, y2 = map(int, best_box.xyxy[0])

    box_width = x2 - x1
    box_height = y2 - y1

    # For a ball, width and height should be close.
    pixel_diameter = (box_width + box_height) / 2.0

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": best_conf,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "cx": cx,
        "cy": cy,
        "box_width": box_width,
        "box_height": box_height,
        "pixel_diameter": pixel_diameter
    }


def main():
    args = parse_args()

    print("[INFO] Loading YOLO model...")
    model = YOLO(args.model, task="detect")

    print("[INFO] Opening ESP32-CAM stream...")
    cap = cv2.VideoCapture(args.stream)

    if not cap.isOpened():
        print("[ERROR] Could not open stream.")
        return

    pixel_samples = []
    recent_values = deque(maxlen=10)

    print()
    print("Calibration instructions:")
    print(f"1. Put the ball exactly {args.known_distance:.1f} cm from the camera.")
    print("2. Keep the ball centered and fully visible.")
    print("3. Press SPACE to record a sample.")
    print("4. Press A to auto-collect samples.")
    print("5. Press Q to quit.")
    print()

    auto_collect = False

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[WARNING] Failed to read frame.")
            continue

        frame = cv2.resize(frame, (args.width, args.height))

        results = model.predict(
            frame,
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False
        )

        detection = get_best_detection(results[0], model)

        display = frame.copy()

        if detection is not None:
            x1 = detection["x1"]
            y1 = detection["y1"]
            x2 = detection["x2"]
            y2 = detection["y2"]
            cx = detection["cx"]
            cy = detection["cy"]

            box_width = detection["box_width"]
            box_height = detection["box_height"]
            pixel_diameter = detection["pixel_diameter"]
            confidence = detection["confidence"]
            class_name = detection["class_name"]

            recent_values.append(pixel_diameter)

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)

            label = (
                f"{class_name} {confidence:.2f} | "
                f"W:{box_width}px H:{box_height}px D:{pixel_diameter:.1f}px"
            )

            cv2.putText(
                display,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display,
                f"Samples: {len(pixel_samples)}/{args.samples}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )

            if auto_collect and len(pixel_samples) < args.samples:
                pixel_samples.append(pixel_diameter)
                time.sleep(0.05)

        else:
            cv2.putText(
                display,
                "No ball detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2
            )

        cv2.putText(
            display,
            "SPACE: sample | A: auto | S: save | Q: quit",
            (10, args.height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.imshow("Ball Calibration", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("a"):
            auto_collect = not auto_collect
            print(f"[INFO] Auto collect: {auto_collect}")

        elif key == ord(" "):
            if detection is not None:
                pixel_samples.append(detection["pixel_diameter"])
                print(
                    f"[SAMPLE] {len(pixel_samples)}/{args.samples} | "
                    f"pixel diameter = {detection['pixel_diameter']:.2f}px"
                )
            else:
                print("[WARNING] No detection to sample.")

        elif key == ord("s"):
            if len(pixel_samples) == 0:
                print("[ERROR] No samples collected yet.")
                continue

            break

        if len(pixel_samples) >= args.samples:
            print("[INFO] Required samples collected.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(pixel_samples) == 0:
        print("[ERROR] Calibration cancelled. No samples saved.")
        return

    avg_pixel_diameter = sum(pixel_samples) / len(pixel_samples)

    focal_length_px = (
        avg_pixel_diameter * args.known_distance
    ) / args.real_diameter

    calibration = {
        "real_ball_diameter_cm": args.real_diameter,
        "known_distance_cm": args.known_distance,
        "known_pixel_diameter_px": avg_pixel_diameter,
        "focal_length_px": focal_length_px,
        "samples_used": len(pixel_samples),
        "frame_width": args.width,
        "frame_height": args.height,
        "imgsz": args.imgsz,
        "model": args.model,
        "stream": args.stream
    }

    with open(args.output, "w") as f:
        json.dump(calibration, f, indent=4)

    print()
    print("[CALIBRATION DONE]")
    print(f"Real ball diameter:      {args.real_diameter:.2f} cm")
    print(f"Known distance:          {args.known_distance:.2f} cm")
    print(f"Average pixel diameter:  {avg_pixel_diameter:.2f} px")
    print(f"Focal length:            {focal_length_px:.2f} px")
    print(f"Saved to:                {args.output}")


if __name__ == "__main__":
    main()
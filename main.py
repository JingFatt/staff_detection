import os
import cv2
import argparse
from ultralytics import YOLO


def run_play_video(video_path, output_path, model_path, conf_threshold=0.35, max_jump=60):

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_id = 0
    prev_cx, prev_cy = None, None

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf_threshold, verbose=False)

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # -------- position filter --------
                if cy < 200:
                    continue

                # -------- temporal smoothing --------
                if prev_cx is not None and abs(cx - prev_cx) > max_jump:
                    continue
                prev_cx, prev_cy = cx, cy

                # -------- draw box --------
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Detection Video", frame)

        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def run_detect(video_path, model_path, crop_folder, scale=3):

    os.makedirs(crop_folder, exist_ok=True)

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)

    # crop_id = 0
    frame_id = 0
    prev_cx, prev_cy = None, None
    max_jump = 60

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.35, verbose=False)

        for r in results:

            if r.boxes is None:
                continue

            for box in r.boxes:

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])

                # -------- position filter --------
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if cy < 200:
                    continue

                # -------- temporal smoothing --------
                if prev_cx is not None and abs(cx - prev_cx) > max_jump:
                    continue
                prev_cx, prev_cy = cx, cy

                print(f"Frame {frame_id} | Staff located at ({cx},{cy}) | conf={conf:.2f}")
                
                # -------- crop detection --------
                crop = frame[y1:y2, x1:x2]
                height, width = crop.shape[:2]
                display_crop = cv2.resize(crop, (width*scale, height*scale))

                save_path = f"{crop_folder}/Frame_{frame_id}_staff.jpg"
                cv2.imwrite(save_path, display_crop)

                cv2.imshow("Detected Staff", display_crop)
                cv2.waitKey(1)

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--play_vid", action="store_true")
    parser.add_argument("--detect", action="store_true")

    parser.add_argument("--video", default="D:/Github Clone/odp/video/sample.mp4")
    parser.add_argument("--output", default="D:/Github Clone/odp/video/output.mp4")
    parser.add_argument("--model", default="D:/Github Clone/odp/runs/detect/output/staff_detection4/weights/best.pt")
    parser.add_argument("--crop_folder", default="D:/Github Clone/odp/output")

    args = parser.parse_args()

    if args.play_vid:

        run_play_video(args.video, args.output, args.model)

    elif args.detect:

        run_detect(args.video, args.model, args.crop_folder)

    else:

        print("Please use --play_vid or --detect")


if __name__ == "__main__":
    main()
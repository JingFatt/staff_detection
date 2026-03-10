import cv2
import os

video_path = "D:/Github Clone/odp/dataset/video/1.mp4"
output_folder = "D:/Github Clone/odp/dataset/pic/1"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_id = 0
save_id = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    if frame_id % 1 == 0:   # 每10 frame存一次
        filename = f"{output_folder}/{save_id:02d}.jpg"
        cv2.imwrite(filename, frame)
        save_id += 1

    frame_id += 1

cap.release()
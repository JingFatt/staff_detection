import cv2
import numpy as np
from ultralytics import YOLO

video_path = 'D:/Github Clone/odp/video/sample.mp4'
output_path = 'D:/Github Clone/odp/video/output.mp4'
model_path = 'D:/Github Clone/odp/model/model.pt'

model = YOLO('yolo26n.pt')

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    frame = results[0].plot()  # Plot the results on the frame

    # Display the resulting frame
    cv2.imshow('Video Player', frame)
    out.write(frame)  # Write the frame to the output video

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
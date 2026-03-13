# Staff Detection System

## Project Description
This project is designed to detect staff members in videos, track their positions, and optionally save cropped images of each detected staff member. The system leverages the YOLOv8 and YOLOv26 models for real-time detection, providing coordinates and confidence scores for each frame. This can be used for security monitoring, staff verification, or attendance systems.

---

## Features
- Detect staff in video frames
- Record coordinates and confidence for each detected staff
- Save cropped staff images
- Output processed video with bounding boxes

---

## Repository Structure
```txt
staff_detection/
│
├─ dataset/ 
├─ output/
├─ video/
├─ .gitignore # Ignore unnecessary files/folders
├─ frame_cap.py
├─ main.py
├─ rename_file.py
├─ tag.py
├─ test.py
├─ train.py
├─ yolo26m.pt
├─ yolo26m-pose.pt


```
---

## Requirements
Install the required Python packages:
```bash
pip install numpy==2.4.3 opencv_python==4.13.0.92 tqdm==4.67.3 ultralytics==8.4.21

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
```

---

## Usage
### Detect staff in a video
```bash
python main.py --detect --video path/to/video.mp4
```
### Save cropped staff images
```bash
python main.py --save_crop --video path/to/video.mp4
```
### Training your own model
```bash
python train.py
```
---

## Target
The goal of this project is to detect this badge (shown as above) and identify the person as a **staff member** (as below).  
<div align="center">
  <img width="193" height="208" alt="tag_piano" src="https://github.com/user-attachments/assets/379d5999-c710-450f-afed-dade3eac019d" />
</div>

<div align="center">
  <img width="193" height="208" alt="01" src="https://github.com/user-attachments/assets/be5ad4e7-2b34-4e6f-b96c-ce9e6bbc4045" />
</div>

## Results
Example output from a test video:
```txt
Frame 423 | Staff located at (528,666) | conf=0.36
Frame 811 | Staff located at (512,649) | conf=0.75
Frame 1160 | Staff located at (506,648) | conf=0.66
```
<div align="center">
  <img src="https://github.com/user-attachments/assets/57b1e3d9-243b-4da3-89f6-e2485aacdb72" width="200"/>
  <img src="https://github.com/user-attachments/assets/1d0de51f-0722-4e58-b07c-53b0af0ea409" width="200"/>
  <img src="https://github.com/user-attachments/assets/a774817d-41c7-4827-b0c0-a5f87be571f1" width="200"/>
</div>




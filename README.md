# Staff Detection System

## Project Description
This project is designed to detect staff members in videos, track their positions, and optionally save cropped images of each staff member detected. The system leverages the [YOLOv26](https://docs.ultralytics.com/models/yolo26/#supported-tasks-and-modes) models for real-time detection, providing coordinates and confidence scores for each frame. This can be used for security monitoring, staff verification, or attendance systems.

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
## Dataset

The dataset for this project was carefully curated from **YouTube videos**, collected from three different sources to ensure diversity in lighting, angles, and environments.  

Frames were extracted from these videos using a Python script, and each frame was processed to embed the **staff badge** onto the corresponding person. These images were then annotated manually to create the **ground truth** for detection.  

The final dataset contains **1,100 images** (after processing), split as follows:  

- **Training set:** 800 images  
- **Validation set:** 200 images  
- **Test set:** 100 images  

This dataset is available for download via Roboflow:  
[Download Staff Badge Dataset](https://app.roboflow.com/ds/00rNnCbH8F?key=9tJY8aODkO)

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

## Model Evaluation

The model performance is measured using **BoxF1**, **Box Precision (BoxP)**, and **Recall (R)**.  
The curves below show how these metrics evolve during training/testing.

### Metrics Curves
<div align="center">
  <img width="400" height="400" alt="BoxR_curve" src="https://github.com/user-attachments/assets/2c48174e-83d6-4cd5-9e68-a103ba891a28" />
  <img width="400" height="400" alt="BoxPR_curve" src="https://github.com/user-attachments/assets/3fd4fe3e-921b-4fce-9f41-1d0b88549663" />
  <img width="400" height="400" alt="BoxP_curve" src="https://github.com/user-attachments/assets/fbc1d3a4-e906-48eb-ac28-9f8beec8daa1" />
  <img width="400" height="400" alt="BoxF1_curve" src="https://github.com/user-attachments/assets/1b9d38a1-6581-496e-aa5c-66d2bf1885c8" />
</div>

### Confusion Matrix
Here is the confusion matrix of the model predictions on the test set:

<div align="center">
  <img width="400" height="400" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/907b8900-6405-4680-8f19-9f4b5a4ea441" />
  <img width="400" height="400" alt="confusion_matrix" src="https://github.com/user-attachments/assets/a3da392c-2d7e-4fef-95f3-48d7675b9d13" />    
</div>
After normalization, the confusion matrix clearly shows the model’s accuracy in detecting **staff members**.

### Test Set Examples with Predictions

Below are examples from the test set showing how accurately the model detects staff members:

| Ground Truth | Prediction |
|--------------|------------|
| <div align="center"><img src="https://github.com/user-attachments/assets/feeed004-7808-480f-97b6-2279e5386e30" width="500"/></div> | <div align="center"><img src="https://github.com/user-attachments/assets/5f7c9c73-e25d-45b5-9328-916985343bd9" width="500"/></div> |
| <div align="center"><img src="https://github.com/user-attachments/assets/79bfc698-0413-45f1-8eb7-b9060bc11061" width="500"/></div> | <div align="center"><img src="https://github.com/user-attachments/assets/1192b5e2-4ffc-4764-9d8c-cf19f4a0ca8f" width="500"/></div> |
| <div align="center"><img src="https://github.com/user-attachments/assets/53555130-b5f5-48cf-9481-2caf54dcbcc5" width="500"/></div> | <div align="center"><img src="https://github.com/user-attachments/assets/7c34690a-7d8f-4abe-93a0-21572cc6981f" width="500"/></div> |

These results demonstrate the model’s capability to correctly detect and identify staff members in the test set.

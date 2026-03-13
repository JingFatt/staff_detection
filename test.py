from ultralytics import YOLO

MODEL_PATH = YOLO("runs/detect/output/staff_detection4/weights/best.pt")
DATASET = "D:/Github Clone/odp/dataset/staff_detection/data.yaml"
DEVICE = 0

def test_model():
    metrics = MODEL_PATH.val(
        data=DATASET,
        split="test",
        device=DEVICE,
        conf=0.25
    )
    return metrics

if __name__ == "__main__":
    test_model()
    print("Testing completed. Check the output for metrics.")
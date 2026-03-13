import os
from ultralytics import YOLO

# -------------------- Config --------------------
DATASET = "D:/Github Clone/odp/dataset/staff_detection/data.yaml"
MODEL_PATH = "yolo26m.pt"
OUTPUT_DIR = "output"

EPOCHS = 100
BATCH_SIZE = 16

DEVICE = 0

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------- Train --------------------
def train_model():

    # print("Using device:", DEVICE)

    model = YOLO(MODEL_PATH)

    model.train(
        data=DATASET,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=8,
        project=OUTPUT_DIR,
        name="staff_detection",
        val=True,
        plots=True
    )


# -------------------- Main --------------------
if __name__ == "__main__":
    train_model()
    print("Training completed. Check the output directory for results.")
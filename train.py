import os
import glob
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# -------------------- Config --------------------
TRAIN_DATA = "D:/Github Clone/odp/dataset/staff_detection/train"
VAL_DATA = "D:/Github Clone/odp/dataset/staff_detection/val"
MODEL_PATH = "yolo26n.pt"
OUTPUT_DIR = "output"
EPOCHS = 50
BATCH_SIZE = 16
VALIDATE_EVERY = 5  # 每训练 5 个 epoch 验证一次
NUM_VAL_IMAGES = 10  # 验证时保存前 N 张可视化图片

BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------- 可视化函数 --------------------
def save_loss_plot(train_losses, val_losses, save_path):
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot([i for i in range(VALIDATE_EVERY, len(train_losses)+1, VALIDATE_EVERY)],
             val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / mAP50")
    plt.legend()
    plt.title("Train & Validation")
    plt.savefig(save_path)
    plt.close()


def save_val_visualization(model, dataset_path, save_dir, num_images=NUM_VAL_IMAGES):
    os.makedirs(save_dir, exist_ok=True)
    images = glob.glob(os.path.join(dataset_path, "images", "*.*"))
    for img_path in images[:num_images]:
        results = model(img_path)
        frame = results[0].plot()
        filename = os.path.basename(img_path)
        cv2.imwrite(os.path.join(save_dir, filename), frame)


# -------------------- Train + Validation --------------------
def train_val(train_dataset, val_dataset, model_path, epochs, batch_size, output_dir):
    model = YOLO(model_path)
    best_map50 = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        print(f"===== Epoch {epoch}/{epochs} =====")

        # ---------------- 训练 ----------------
        model.train(
            data=os.path.join(train_dataset, "data.yaml"),
            epochs=1,
            batch=batch_size
        )

        # 获取训练 loss
        epoch_train_loss = model.history[-1]['box_loss'] + \
                           model.history[-1]['cls_loss'] + \
                           model.history[-1]['obj_loss']
        train_losses.append(epoch_train_loss)
        print(f"Epoch {epoch} Train Loss: {epoch_train_loss:.4f}")

        # ---------------- 验证 ----------------
        if epoch % VALIDATE_EVERY == 0:
            metrics = model.val(
                data=os.path.join(val_dataset, "data.yaml"),
                batch=batch_size
            )
            val_map50 = metrics.box.map50
            val_losses.append(val_map50)
            print(f"Validation mAP50: {val_map50}")

            # 保存验证集可视化图片
            val_vis_dir = os.path.join(output_dir, "val_visualizations")
            save_val_visualization(model, val_dataset, val_vis_dir)
            print(f"Validation images saved in {val_vis_dir}")

            # 保存最佳模型
            if val_map50 > best_map50:
                best_map50 = val_map50
                model.save(BEST_MODEL_PATH)
                print(f"New best model saved: {BEST_MODEL_PATH}")

        # ---------------- 保存 loss plot ----------------
        loss_plot_path = os.path.join(output_dir, "loss_plot.png")
        save_loss_plot(train_losses, val_losses, loss_plot_path)


# -------------------- Main --------------------
if __name__ == "__main__":
    train_val(TRAIN_DATA, VAL_DATA, MODEL_PATH, EPOCHS, BATCH_SIZE, OUTPUT_DIR)
    print("Training and validation completed. Best model saved at:", BEST_MODEL_PATH)
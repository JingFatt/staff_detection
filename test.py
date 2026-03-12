import random
import cv2
import glob
import argparse
import os
from ultralytics import YOLO

DEFAULT_TRAIN_PATH = "D:/Github Clone/odp/dataset/staff_detection/train"
DEFAULT_VAL_PATH = "D:/Github Clone/odp/dataset/staff_detection/val"
DEFAULT_TEST_PATH = "D:/Github Clone/odp/dataset/staff_detection/test"

DEFAULT_MODEL_PATH = "yolo26n.pt"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 4
DEFAULT_VALIDATION_EVERY = 5
DEFAULT_NUM_CHECK_IMAGES = 5

BEST_MODEL_PATH = os.path.join(DEFAULT_OUTPUT_DIR, "best_model.pt")

# -------------------- Utility Functions --------------------
def get_image_paths(dataset_path):
    return glob.glob(os.path.join(dataset_path, "images", "*.*"))

def get_bounding_box(img, label_path):
    h, w = img.shape[:2]
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                _, x, y, bw, bh = map(float, line.split())

                x1 = int((x - bw/2) * w)
                y1 = int((y - bh/2) * h)
                x2 = int((x + bw/2) * w)
                y2 = int((y + bh/2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return img

def data_visualization(img, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, filename), img)

# -------------------- Main Functions --------------------
def check_dataset(dataset_path, num_images=DEFAULT_NUM_CHECK_IMAGES):
    images = get_image_paths(dataset_path)
    print(f"Found {len(images)} images in {dataset_path}.")

    for _ in range(num_images):

        img_path = random.choice(images)
        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")

        img = cv2.imread(img_path)
        img = get_bounding_box(img, label_path)
        cv2.imshow("Dataset Check", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

def train(dataset_path, model_path=DEFAULT_MODEL_PATH, epochs=DEFAULT_EPOCHS,
          batch_size=DEFAULT_BATCH_SIZE, output_dir=DEFAULT_OUTPUT_DIR):

    model = YOLO(model_path)
    train_output = os.path.join(output_dir, "train_results")
    os.makedirs(train_output, exist_ok=True)
    model.train(data=os.path.join(dataset_path, "data.yaml"), epochs=epochs, 
                batch=batch_size, imgsz=640, project=train_output, 
                name="training")

def val(dataset_path, model_path=DEFAULT_MODEL_PATH, batch_size=DEFAULT_BATCH_SIZE, 
        output_dir=DEFAULT_OUTPUT_DIR):

    model = YOLO(model_path)
    val_output = os.path.join(output_dir, "val_results")
    os.makedirs(val_output, exist_ok=True)
    metrics = model.val(data=os.path.join(dataset_path, "data.yaml"), batch=batch_size, 
                        imgsz=640, project=val_output, name="validation")
    print(metrics)

    # 保存可视化图片
    images = get_image_paths(dataset_path)
    for img_path in images[:10]:  # 只保存前10张监控图片，避免太多
        results = model(img_path)
        frame = results[0].plot()
        data_visualization(frame, val_output, os.path.basename(img_path))
    print(f"Validation completed. Metrics saved to {val_output}")
    print(f"Validation images saved in {val_output}")

def test(dataset_path, model_path=DEFAULT_MODEL_PATH, output_dir=DEFAULT_OUTPUT_DIR):

    model = YOLO(model_path)
    test_output = os.path.join(output_dir, "test_results")
    os.makedirs(test_output, exist_ok=True)

    images = get_image_paths(dataset_path)
    for img_path in images:
        results = model(img_path)
        frame = results[0].plot()
        data_visualization(frame, test_output, os.path.basename(img_path))
    print(f"Testing completed. Results saved to {test_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Staff Detection")

    parser.add_argument("--check", action="store_true", help="Visualize and check dataset")
    parser.add_argument("--train", action="store_true", help="Train model on dataset")
    parser.add_argument("--val", action="store_true", help="Validate model on dataset")
    parser.add_argument("--test", action="store_true", help="Run inference on dataset")

    parser.add_argument("--train_dataset", type=str, default=DEFAULT_TRAIN_PATH, help="Training dataset path")
    parser.add_argument("--val_dataset", type=str, default=DEFAULT_VAL_PATH, help="Validation dataset path")
    parser.add_argument("--test_dataset", type=str, default=DEFAULT_TEST_PATH, help="Test dataset path")

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="YOLO model path")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory for test results")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for training/validation")
    parser.add_argument("--num_check", type=int, default=DEFAULT_NUM_CHECK_IMAGES, help="Number of images to check in dataset")

    args = parser.parse_args()

    if args.check:
        check_dataset(args.dataset, num_images=args.num_check)
    if args.train:
        train(args.train_dataset, model_path=args.model, epochs=args.epochs,
              batch_size=args.batch_size, output_dir=args.output)
    if args.val:
        val(args.val_dataset, model_path=args.model, batch_size=args.batch_size,
            output_dir=args.output)
    if args.test:
        test(args.test_dataset, model_path=args.model, output_dir=args.output)
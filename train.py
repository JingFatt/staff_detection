from ultralytics import YOLO

# import os
# import random
# import cv2

# image_dir = "D:/dataset/images"
# label_dir = "D:/dataset/labels"

# image_name = random.choice(os.listdir(image_dir))
# image_path = os.path.join(image_dir, image_name)
# label_path = os.path.join(label_dir, image_name.replace(".jpg", ".txt").replace(".png", ".txt"))

# img = cv2.imread(image_path)
# h, w = img.shape[:2]

# with open(label_path, "r") as f:
#     lines = f.readlines()

# for line in lines:
#     cls, x, y, bw, bh = map(float, line.split())
#     x1 = int((x - bw/2) * w)
#     y1 = int((y - bh/2) * h)
#     x2 = int((x + bw/2) * w)
#     y2 = int((y + bh/2) * h)

#     cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
#     cv2.putText(img, "staff", (x1, y1-5),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    
# cv2.imshow("Dataset Check", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

model = YOLO(yolo)

2
1
1


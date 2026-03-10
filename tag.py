from ultralytics import YOLO
import cv2
import os
import random
import numpy as np
from tqdm import tqdm

# models
person_model = YOLO("yolo26m.pt")
pose_model = YOLO("yolo26m-pose.pt")

image_folder = "D:/Github Clone/odp/dataset/raw"
output_folder = "D:/Github Clone/odp/dataset/new"
tag_path = "D:/Github Clone/odp/dataset/tag_piano.png"

os.makedirs(output_folder, exist_ok=True)

tag_original = cv2.imread(tag_path)

def random_motion_blur(img):

    k = random.choice([3,5,7])
    kernel = np.zeros((k,k))
    kernel[int((k-1)/2), :] = np.ones(k)
    kernel = kernel / k

    return cv2.filter2D(img, -1, kernel)

def perspective_tag(tag):

    h, w = tag.shape[:2]

    src = np.float32([
        [0,0],
        [w,0],
        [w,h],
        [0,h]
    ])

    shift = 0.25

    dst = np.float32([
        [random.uniform(0,w*shift), random.uniform(0,h*shift)],
        [w-random.uniform(0,w*shift), random.uniform(0,h*shift)],
        [w-random.uniform(0,w*shift), h-random.uniform(0,h*shift)],
        [random.uniform(0,w*shift), h-random.uniform(0,h*shift)]
    ])

    M = cv2.getPerspectiveTransform(src,dst)

    warped = cv2.warpPerspective(tag,M,(w,h))

    return warped

image_list = os.listdir(image_folder)
for img_name in tqdm (image_list, desc="Processing images"):

    img_path = os.path.join(image_folder,img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    results = pose_model(img, verbose=False)

    person_boxes = []
    for r in results:

        if r.keypoints is None:
            continue

        keypoints = r.keypoints.xy

        for person in keypoints:

            # shoulders index (COCO)
            left_shoulder = person[5]
            right_shoulder = person[6]

            if left_shoulder[0]==0 or right_shoulder[0]==0:
                continue
            person_boxes.append(person)
    
    if len(person_boxes)==0:
        continue

    num_to_tag = min(len(person_boxes), random.randint(1,3))
    selected_people = random.sample(person_boxes, num_to_tag)

    for person in selected_people:
        left_shoulder = person[5]
        right_shoulder = person[6]
    
        chest_x = int((left_shoulder[0] + right_shoulder[0]) / 2 + 5)
        chest_y = int((left_shoulder[1] + right_shoulder[1]) / 2 + 20)

        shoulder_dist = abs(left_shoulder[0] - right_shoulder[0])

        tag_w = max(int(shoulder_dist * random.uniform(0.25,0.35)), 10)
        tag_h = max(int(tag_w * 0.6), 6)

        tag = cv2.resize(tag_original,(tag_w,tag_h))

        if random.random()<0.8:
            tag = perspective_tag(tag)

            if random.random()>0.5:
                tag = random_motion_blur(tag)

            alpha = random.uniform(0.5,1.2)
            tag = cv2.convertScaleAbs(tag,alpha=alpha)

            # noise = np.random.normal(0,5,tag.shape).astype(np.uint8)
            # tag = cv2.add(tag,noise)
        
        else:
            tag = tag.copy()

        x1 = chest_x - tag_w//2
        y1 = chest_y

        if x1<0 or y1<0:
            continue

        if x1+tag_w > img.shape[1] or y1+tag_h > img.shape[0]:
            continue

        img[y1:y1+tag_h, x1:x1+tag_w] = tag

    cv2.imwrite(os.path.join(output_folder,img_name),img)
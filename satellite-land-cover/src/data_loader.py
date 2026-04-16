
import os
import cv2
import numpy as np

def load_images(data_dir, limit_per_class=200, img_size=(64, 64)):
    images = []
    labels = []

    for cls in os.listdir(data_dir):
        class_path = os.path.join(data_dir, cls)

        if not os.path.isdir(class_path) or cls == "allBands":
            continue

        count = 0

        for img_name in os.listdir(class_path):
            if count >= limit_per_class:
                break

            img_path = os.path.join(class_path, img_name)

            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)

            images.append(img)
            labels.append(cls)

            count += 1

    return np.array(images), np.array(labels)

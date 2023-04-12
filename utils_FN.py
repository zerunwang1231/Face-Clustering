from os import listdir
from os.path import isdir
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2

# create the detector, using default weights
mtcnn = MTCNN(image_size = 120,min_face_size=50, keep_all=True, thresholds = [0.7, 0.8, 0.8],device="cuda").eval()
model = InceptionResnetV1(pretrained='casia-webface').eval()

# extract a single face from a given photograph
def draw_bbox(bounding_boxes, image):
    for i in range(len(bounding_boxes)):
        x1, y1, x2, y2 = bounding_boxes[i]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)),
                     (0, 0, 255), 2)
    return image

def face_boxes(image):
    detected_boxes = mtcnn.detect(image, landmarks=False)
    return detected_boxes

def face_landmarks_encodings(image):
    descriptors = []
    ims = []
    img_cropped = mtcnn(image)
    if img_cropped is not None:
        for item in img_cropped:
            # get embedding
            img_probs = model(item.unsqueeze(0)).tolist()
            descriptors.append(img_probs)
            ims.append((image, item.tolist()))
    return descriptors, ims


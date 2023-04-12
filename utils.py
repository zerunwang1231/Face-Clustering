import dlib
import cv2
import numpy as np
from pathlib import Path

cur_dir = Path(__file__).parent.absolute()
detector = dlib.cnn_face_detection_model_v1(str(cur_dir/"external_models/mmod_human_face_detector.dat"))
shape_predictor = dlib.shape_predictor(str(cur_dir/"external_models/shape_predictor_68_face_landmarks.dat"))
encoder = dlib.face_recognition_model_v1(str(cur_dir/"external_models/dlib_face_recognition_resnet_model_v1.dat"))

def process_boxes(box):
    xmin = box.rect.left()
    ymin = box.rect.top()
    xmax = box.rect.right()
    ymax = box.rect.bottom()
    return [int(xmin), int(ymin), int(xmax), int(ymax)]

def face_boxes(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detected_boxes = detector(image_rgb, 1)
    return detected_boxes


def face_landmarks_encodings(image):
    # compute the facial embeddings for each face 
    # in the input image. the `compute_face_descriptor` 
    # function returns a 128-d vector that describes the face in an image
    descriptors = []
    ims = []
    for face_box in face_boxes(image):
        face_box_rect = face_box.rect # change type from mmod_rect to rect
        shape = shape_predictor(image, face_box_rect)
        descriptors.append(encoder.compute_face_descriptor(image, shape))
        ims.append((image, shape))
    return descriptors, ims

# reference: https://dontrepeatyourself.org/post/face-recognition-with-python-dlib-and-deep-learning/
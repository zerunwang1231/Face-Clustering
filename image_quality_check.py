from face_image_quality import SER_FIQ
import cv2
from pathlib import Path
from os import listdir,makedirs
from os.path import isfile, join
import utils
import pickle
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
import numpy as np


ser_fiq = SER_FIQ(gpu=None) # my cudo version does not support mxnet
cur_dir = Path(__file__).parent.absolute()
image_files_label0 = [
        f for f in listdir(cur_dir / "output_chips/0") if isfile(join(cur_dir /"output_chips/0", f))
    ]
image_files_label0.sort()
scores = {}
# using CPU, only try with a few images
for f in image_files_label0:
    print("Processing file: {}".format(f))
    image = cv2.imread(join(cur_dir / "output_chips/0", f))
    aligned_img = ser_fiq.apply_mtcnn(image)
    if aligned_img is not None:
        score = ser_fiq.get_score(aligned_img, T=100)
        scores[f] = score
        print("SER-FIQ quality score of face",f, "is", score)
print(scores)




# reference : https://github.com/pterhoer/FaceImageQuality
# Paper:SER-FIQ: Unsupervised Estimation of Face Image Quality Based on
# Stochastic Embedding Robustness  https://arxiv.org/pdf/2003.09373.pdf
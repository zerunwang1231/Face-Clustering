from pathlib import Path
from os import listdir,makedirs
from os.path import isfile, join
import utils_FN
import cv2
import pickle
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import time
import glob
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from PIL import Image
import numpy as np

def main():
    cur_dir = Path(__file__).parent.absolute()
    image_files = [
        f for f in listdir(cur_dir / "event26") if isfile(join(cur_dir / "event26", f))
    ]
    # set min_face_size to be large, avoid detecting non face
    mtcnn = MTCNN(min_face_size=50, keep_all=True,thresholds = [0.7, 0.8, 0.8], device="cuda")
    # uncomment below to conduct face detecting
    """
    for f in image_files:
        print("Processing file: {}".format(f))
        save_name = join(cur_dir / "detect_result_FN", f)
        image = cv2.imread(join(cur_dir / "event26", f), cv2.COLOR_BGR2RGB)
        detected_boxes, conf= mtcnn.detect(image, landmarks=False)
        if detected_boxes is not None:
            image = utils_FN.draw_bbox(detected_boxes, image)        
        cv2.imshow('Result', image/255.0)
        cv2.waitKey(1)
        #cv2.waitKey(0)
        if not cv2.imwrite(save_name, image):
            raise Exception("Could not write image")
        if detected_boxes is not None:
            print(f"Total faces detected: {len(detected_boxes)}")
        else:
            print("No face detected")
    """

    # The model does worse in dertecting faces compared to the one from dlib.
    # It is non-sensitive to side faces and it has larger bias (i.e., it detect non-faces very often compared to dlib)

    # embedding
    path1 = Path(cur_dir/"encodings_FN/descriptors.pickle")
    path2 = Path(cur_dir/"encodings_FN/ims.pickle")
    if not path1.is_file():
        descriptors = []
        ims = []
        for f in image_files:
            print("Processing file: {}".format(f))
            image = Image.open(join(cur_dir / "event26", f))
            descriptors_sub, ims_sub = utils_FN.face_landmarks_encodings(image)
            descriptors.extend(descriptors_sub)
            ims.extend(ims_sub)
            torch.cuda.empty_cache() 
        # save objects
        with open(cur_dir/"encodings_FN/descriptors.pickle", "wb") as f:
            pickle.dump(descriptors, f)
        with open(cur_dir/"encodings_FN/ims.pickle", "wb") as f:
            pickle.dump(ims, f)
    elif not path1.is_file() or not path2.is_file():
        raise Exception("Remove the file")
    else:
        # read descriptors and ims
        with open(cur_dir/"encodings_FN/descriptors.pickle", "rb") as f:
            descriptors = pickle.load(f)
        with open(cur_dir/"encodings_FN/ims.pickle", "rb") as f:
            ims = pickle.load(f)



    descriptor_re = []
    # refreme descriptor
    for i in descriptors:
        descriptor_re.extend(i)
        # clustering
    labels = AgglomerativeClustering(linkage="ward",n_clusters = 70).fit(descriptor_re).labels_
    num_classes = len(set(labels))
    print("Number of clusters: {}".format(num_classes))


    face_dict = defaultdict(list)
    for i in range(len(labels)):
        face_dict[labels[i]].append(ims[i])

    for key in face_dict.keys():
            file_dir1 = join(cur_dir/"output_chips_FN",str(key))
            file_dir2 = join(cur_dir/"output_images_FN",str(key))
            if not Path(file_dir1).is_file():
                makedirs(file_dir1)
            if not Path(file_dir2).is_file():
                makedirs(file_dir2)
            for index, (image,shape) in enumerate(face_dict[key]):
                # save face chips
                file_path1 = join(file_dir1, "face_"+str(index)+".jpg")
                img = np.array(shape)[0]*255
                img = Image.fromarray(img).convert('RGB')
                img.save(fp = Path(file_path1))
                # save full image
                file_path2 = join(file_dir2, "image_"+str(index)+".jpg")
                image = image.convert('RGB')
                image.save(fp = Path(file_path2))









if __name__ == "__main__":
    main()


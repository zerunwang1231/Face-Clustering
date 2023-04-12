import cv2
from pathlib import Path
from os import listdir,makedirs
from os.path import isfile, join
import utils
import pickle
import dlib
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering

def main():
    cur_dir = Path(__file__).parent.absolute()
    image_files = [
        f for f in listdir(cur_dir / "event26") if isfile(join(cur_dir / "event26", f))
    ]
    # face detection;
    # Use CNN detector rather than HOG to increase accuracy
    # Besides, CNN method under dlib can use GPU. The prerequisite is cuda cudnn package
    # detect faces in each image, save images with result in file detect_result
    print(dlib.DLIB_USE_CUDA) # check whether CUDA is in use
    """
    for f in image_files:
        print("Processing file: {}".format(f))
        save_name = join(cur_dir / "detect_result", f)
        image = cv2.imread(join(cur_dir / "event26", f))
        detected_boxes = utils.face_boxes(image)
        for box in detected_boxes:
            res_box = utils.process_boxes(box)
            cv2.rectangle(image, (res_box[0], res_box[1]),
                  (res_box[2], res_box[3]), (0, 255, 0), 
                  2)
        cv2.imshow('Result', image)
        cv2.waitKey(1)
        if not cv2.imwrite(save_name, image):
            raise Exception("Could not write image")
        print(f"Total faces detected: {len(detected_boxes)}")
    """
    # face encoding
    path1 = Path(cur_dir/"encodings/descriptors.pickle")
    path2 = Path(cur_dir/"encodings/ims.pickle")
    if not path1.is_file() and not path2.is_file():
        descriptors = []
        ims = []
        for f in image_files:
            print("Processing file: {}".format(f))
            image = cv2.imread(join(cur_dir / "event26", f))
            descriptors_sub, ims_sub = utils.face_landmarks_encodings(image)
            descriptors.extend(descriptors_sub)
            ims.extend(ims_sub)
        # save objects
        with open(cur_dir/"encodings/descriptors.pickle", "wb") as f:
            pickle.dump(descriptors, f)
        with open(cur_dir/"encodings/ims.pickle", "wb") as f:
            pickle.dump(ims, f)
    elif not path1.is_file() or not path2.is_file():
        raise Exception("Remove the file")
    else:
        # read descriptors and ims
        with open(cur_dir/"encodings/descriptors.pickle", "rb") as f:
            descriptors = pickle.load(f)
        with open(cur_dir/"encodings/ims.pickle", "rb") as f:
            ims = pickle.load(f)

        """
        # use the dlib.chinese_whispers_clustering() for clustering
        labels = dlib.chinese_whispers_clustering(descriptors, 0.5)
        num_classes = len(set(labels))
        print("Number of clusters: {}".format(num_classes))
        # under chinese_whispers_clustering, we have in total 21 clusters with threshold 0.5. 
        # Some of clusters have satisfying result, i.e., cluster the same person into one label, such as label 2 (folder 2)
        # Some clusrters give acceptable classicafication, i.e., mistake someone else's face as the current person,
        # such as label 1 (folder 1)
        # Some clusters fail to make meaningful classification,i.e., they include all face chips that are deviated from 
        # other identified labels, such as label 0 (folder 0) that contains mostly female faces and label 5 (folder 5) that 
        # includes mostly side faces.
        # Such problem may due to small cluster numbers (equivlently, the threshold is too large)
        # We may  improve it with a smaller threshold, but the optimal value is very time-consuming to determine
        # Therefore, we switch to use another method, where we can specify the number of clusters. (dlib.chinese_whispers_clustering
        # only intakes threshold)
        """
        # We try to use a method that is designed for a large cluster number
        # We implement hierarchical clustering as an alternative clustering method
        # n_clusters is estimated based on the dataset
        labels = AgglomerativeClustering(linkage="ward",n_clusters = 70).fit(descriptors).labels_
        num_classes = len(set(labels))
        print("Number of clusters: {}".format(num_classes))


        face_dict = defaultdict(list)
        for i in range(len(labels)):
            face_dict[labels[i]].append(ims[i])
        """
        for key in face_dict.keys():
            file_dir1 = join(cur_dir/"output_chips",str(key))
            file_dir2 = join(cur_dir/"output_images",str(key))
            if not Path(file_dir1).is_file():
                makedirs(file_dir1)
            if not Path(file_dir2).is_file():
                makedirs(file_dir2)
            for index, (image,shape) in enumerate(face_dict[key]):
                # save face chips
                file_path1 = join(file_dir1, "face_"+str(index))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                dlib.save_face_chip(image_rgb, shape, file_path1, size = 150, padding = 0.25)
                # save full image
                file_path2 = join(file_dir2, "image_"+str(index)+".jpg")
                if not cv2.imwrite(file_path2, image):
                    raise Exception("Could not write image")
    # reference: http://dlib.net/face_clustering.py.html
            """












if __name__ == "__main__":
    main()

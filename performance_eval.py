import dlib
import cv2
from pathlib import Path
import pickle
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from collections import defaultdict
from os import makedirs
from os.path import join


# read descriptors under dlib and facenet
cur_dir = Path(__file__).parent.absolute()
with open(cur_dir/"encodings/descriptors.pickle", "rb") as f:
    descriptors = pickle.load(f)
with open(cur_dir/"encodings_FN/descriptors.pickle", "rb") as f:
    descriptors_orig = pickle.load(f)

descriptors_FN = []
# refreme descriptor
for i in descriptors_orig:
    descriptors_FN.extend(i)

labels_ac = AgglomerativeClustering(linkage="ward",n_clusters = 70).fit(descriptors).labels_
labels_ac_FN = AgglomerativeClustering(linkage="ward",n_clusters = 70).fit(descriptors_FN).labels_
labels_cwc = dlib.chinese_whispers_clustering(descriptors, 0.5)

# Silhouette score (desirable: close to 1)
ss_ac = silhouette_score(descriptors, labels_ac)
print("The Silhouette score for Agglomerative Clustering with Ward linkage (dlib): %f" % ss_ac) # 0.142462
# This result shows that the cluster is valid, but the elements in each cluster may not be very dense
ss_cwc = silhouette_score(descriptors, labels_cwc)
print("The Silhouette score for Chinese whisper algorithm (dlib): %f" % ss_cwc) # 0.065650
# Agglomerative Clustering and Chinese whisper algorithm under dlib  have the same input. Thus this comparesion shows which cluster
# is better for the given input  
# Also provide the score for FaceNet for reference
ss_ac_FN = silhouette_score(descriptors_FN, labels_ac_FN)
print("The Silhouette score for Agglomerative Clustering with Ward linkage (FaceNet): %f" % ss_ac_FN) # 0.130452
# We may compare it with ss_ac to see which model gives better cluster results under Agglomerative Clustering, if we only care
# about whether elements within each cluster are uniform and across each cluster are distinct  

# Calinski Harabasz Score (desirable: high value)
chs_ac = calinski_harabasz_score(descriptors, labels_ac)
print("The Calinski Harabasz score for Agglomerative Clustering with Ward linkage (dlib): %f" % chs_ac) # 65.329357
chs_cwc = calinski_harabasz_score(descriptors, labels_cwc)
print("The Calinski Harabasz score for Chinese whisper algorithm (dlib): %f" % chs_cwc) # 78.184895
# Chinese whisper algorithm gives highest ratio of var(across cluster)/var(within cluster)
chs_ac_FN = calinski_harabasz_score(descriptors_FN, labels_ac_FN)
print("The Calinski Harabasz score for Agglomerative Clustering with Ward linkage (FaceNet): %f" % chs_ac_FN) # 54.476561


# The above scores aspires us to select a good threshold value / number of cluster for our clustering 
# hyperparameter tuning taking Silhouette Score as banchmark
# We do not use Calinski Harabasz Score becasue we observe that a lower cluster number leads to higher CH score in practice, which 
# is not suitable for our case.

threshold = [0.4, 0.45, 0.5, 0.55] # parameter more than 0.6 cannot be executed in our case
n_clusters = range(40, 71)
score_ac = []
score_cwc = []
for n in n_clusters:
    labels_ac = AgglomerativeClustering(linkage="ward",n_clusters = n).fit(descriptors).labels_
    ss_ac = silhouette_score(descriptors, labels_ac)
    score_ac.append(ss_ac)
print("Scores for Agglomerative Clustering (dlib):")
print(score_ac)
print("Selected parameter: %i" %n_clusters[score_ac.index(max(score_ac))]) # 69
print("The highest score is: %f" % max(score_ac)) # 0.143520
for t in threshold:
    labels_cwc = dlib.chinese_whispers_clustering(descriptors, t)
    ss_cwc = silhouette_score(descriptors, labels_cwc)
    score_cwc.append(ss_cwc)
print("Scores for Chinese Whispers Clustering (dlib):")
print(score_cwc)
print("Selected parameter: %f" % threshold[score_cwc.index(max(score_cwc))]) # 0.4
print("The highest score is: %f" % max(score_cwc)) #  0.1638755
labels_cwc = dlib.chinese_whispers_clustering(descriptors, 0.4)
num_classes = len(set(labels_cwc))
print("Number of clusters: {}".format(num_classes)) # 540
# The number of clusters is unreasonably high. Therefore, we do not use this result for final image clustering


"""
# Classify images based on updated parameters:
with open(cur_dir/"encodings/ims.pickle", "rb") as f:
    ims = pickle.load(f)
# Agglomerative Clustering
labels_ac = AgglomerativeClustering(linkage="ward",n_clusters = 69).fit(descriptors).labels_
face_dict_ac = defaultdict(list)
for i in range(len(labels_ac)):
    face_dict_ac[labels_ac[i]].append(ims[i])
for key in face_dict_ac.keys():
    file_dir1 = join(cur_dir/"output_chips_refine",str(key))
    file_dir2 = join(cur_dir/"output_images_refine",str(key))
    if not Path(file_dir1).is_file():
        makedirs(file_dir1)
    if not Path(file_dir2).is_file():
        makedirs(file_dir2)
    for index, (image,shape) in enumerate(face_dict_ac[key]):
        # save face chips
        file_path1 = join(file_dir1, "face_"+str(index))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dlib.save_face_chip(image_rgb, shape, file_path1, size = 150, padding = 0.25)
        # save full image
        file_path2 = join(file_dir2, "image_"+str(index)+".jpg")
        if not cv2.imwrite(file_path2, image):
            raise Exception("Could not write image")
"""

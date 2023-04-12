Folders:
even26: dataset
externel_models: downloaded from dlib documents for building detectors and other necessary objects under dlib.
detect_result: face detection results using CNN method from dlib
detect_result_FN: face detection result using facenet
encodings: face encodings by dlib
encodings_FN: face encodings by facenet
output_chips_cwc: face chaips classification using dlib.chinese_whispers_clustering()
output_imagess_cwc: image classification using dlib.chinese_whispers_clustering(), one to one with output_chips_cwc
output_chips: face chaips classification using AgglomerativeClustering() 
output_images: image classification using AgglomerativeClustering(), one to one with output_chips
output_chips_refine: face chaips classification using AgglomerativeClustering(). Hyperparamter n_clusters is tuned based on Silhouette score
output_images_refine: image classification using AgglomerativeClustering(), one to one with output_chips. Hyperparamter n_clusters is tuned based on Silhouette score
output_chips_FN: face chaips classification using AgglomerativeClustering() and facenet
output_images_FN: image classification using AgglomerativeClustering() and facenet, one to one with output_chips 



Python files:
main.py:  main python file using dlib. 
utils.py: contains all functions based on dlib
main_FN.py:  main python file using facenet. 
utils_FN.py: contains all functions based on facenet
performance_eval.py: performance evaluation


Summary 

dlib package:
In face detection, we implement CNN method, which has better performance than its alternative, HOG. The model shows good performace in general, especially when there are only a few people in the photo (e.g., 1~30). However, it may fail to detect faces when there are dozens of people (e.g., more than 50) in the picture (e.g., refer to 430.jpg in detect_result), perhaps due to resolution

In face clustering, we have tried two method. The first one is dlib.chinese_whispers_clustering(). With threshold set to be 0.5, this method gives 21 clusters (which is smaller than the ideal number), and the performance within each cluster varies. Some of clusters have satisfying result, while some do not. Such problem occurs because our threshold is too large (thus clusters are small, and variance-within is large). We may improve it by setting a smaller threshold or a larger clusrer number. However, dlib.chinese_whispers_clustering() does not intake preset cluster number and a optimal selection of threshold is hard to make, since we cannot quantify our result with our dataset. 

Therefore, we wish to select clusters manually, while take the algorithm that minimize variance within each cluster. We implement AgglomerativeClustering() with cluster number 70. The number is selected by estimating how many different people in the whole dataset (e.g., refer to 430.JPG). In this case, we observe a better clustering result. Most clusters (refer to each folder under directory output_chips) have small variance (aka, cluster the same person into one label). However, we still have a few clusters that do not perform well (such as label 10), but the overall performance seems to be better than chinese_whispers_clustering(). (I personally thinkn this clustering does better job than than me)

FaceNet Packages:
The given mtcnn model is less sensitive to side faces / low resolution faces when it comes to face detection. Besides, it has larger bias compared to dlib (i.e., it mistaken non-face object as faces more frequently and it fails to make a detection sometimes compared to dlib). However, on my device, with the same GPU accelerator, it takes less time for FaceNet to process an image compared to dlib. Note that this metod mainly detects high quality face images (e.g., front faces and side faces with high resolution) and a few non-face objects. Therefore it should be easier for clustering algorthem to label the embedded face chips compared to the case in dlib. However, since mtcnn in general detect less faces than dlib, the overall accuracy is lower than dlib.

Accuracy (Performance):
For the unlabeled dataset, it is hard to caculate the actual accuracy of the dataset, where we need true lables. However, it is possible to compare the performance based on the entropy or other similar measure. We want our clusters to have low entropy (less chaos) within a cluster, but have higher entropy across different clusters (each clusters are distinct). Possible approaches includes Silhouette score and Calinski Harabasz Score. Silhouette score is based on the comparsion of the average distance of a data within its cluster and the min distance of it to the other cluster. Note that this method cannot be applied to concentric data (e.g., dataset for DBscan). A desired cluster should give a score that is close to 1, or at least larger than 0. Calinski Harabasz Score utilizes variance instead, and it is the ratio of average vairance across clusters and average variance witin each clusters. A desired cluster should give a higher score among its competitors.
We implement both method in file performance_eval.py. For models in main.py, under dlib, Chinese Whispers Clustering (CWC) has a lower Silhouette (S) score but a higher Calinski Harabasz (CH) score comnpared to  Agglomerative Clustering (AC). Meawhile, under AC with same cluster numbers, embeddings given by dlib have higher S score compared to those given by FaceNet. However, in practice we notice that usually CH score increases when the number of clusters decrease, while S score tends to increase when the number of cluster increases. For example, with our embeddings, CH score will perfer a thereshold of 0.55 among [0.4, 0.5, 0.55] for CWC, which results in 4 clusters only. However, S score will perfer a threshold of 0.4, which results in 560 clusters. Both methods are not reliable for tuning parameters for CWC. We try to use S score to tune number of clusters from range 50 to 70 for AC, and 69 is selected. The performance imporvement is not significant. We reset parameter to be 69 for AC and conduct image classfication upon the results and save the updated face chips and images to output_chips_refine and output_images_refine. 


Face Quality Check:
Due to short of time, this part is not completed. However, I sucessfully run a face quality check model from this paper https://arxiv.org/pdf/2003.09373.pdf
I implement this model to the face chips from /output_chips/0. Please refer to image_quality_check.py for details. 

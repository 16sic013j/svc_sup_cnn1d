from sklearn.datasets import fetch_20newsgroups

from Baseline.sklearn_cluster import kmeans, dbscan
from Clustering.faiss import build_faiss_knns
from Clustering.gcndcluster import generate_cluster, save_proposals
from Clustering.similarysearch import getnormlize_features
from Evaluation.evaluate import evaluation
from FeatureExtract.featureextract import extractfeature
from Preprocess.preprocess import preprocess, label2file
from utils import writepred, ifexitdir, np

feature_dir = "FeatureExtract"
knn_dir = "Clustering/knn"
proposal_dir = "Clustering/proposal"
dataset_dir = "Dataset"

# path for offline datasets
# labelfile_path = dataset_dir + "labelfile.txt"
# data_preprocessed_file = "dataset_dir + "raw_data.txt"

labelfile_path = "Preprocess/labelfile.txt"
data_preprocessed_file = "Preprocess/data_preprocessed.txt"
svertex_predlabelfile = "Clustering/proposal/svertex_predlabelfile.txt"
knn_predlabelfile = "Baseline/knn_predlabels.txt"
dbscan_predlabelfile = "Baseline/dbscan_predlabels.txt"
modelpath = feature_dir + '/model.bin'
featurefilepath = feature_dir + '/featurefile.npy'
uniquetokenfile = feature_dir + "/unniquetokens.txt"
glovefile = feature_dir + '/glove.6B.100d.txt'

# Reset all generated directory and files for retests
dirarr = [knn_dir, proposal_dir]
filearr = [modelpath, featurefilepath]
ifexitdir(dirarr, filearr)

# # Gather dataset and add to memory
# newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
# data_raw = newsgroups_train.data
# labels = newsgroups_train.target
# classnames = newsgroups_train.target_names

# # Preprocess datasets
# preprocess(data_raw, data_preprocessed_file)
# label2file(labels, labelfile_path)

# # Extract feature
# classlen = len(classnames)
# MAX_SEQUENCE_LENGTH = 1000
# MAX_NB_WORDS = 20000
# EMBEDDING_DIM = 100
# VALIDATION_SPLIT = 0.2
# extractfeature(data_preprocessed_file,
#                labelfile_path,
#                classlen,glovefile,
#                uniquetokenfile,
#                modelpath,
#                featurefilepath,
#                MAX_SEQUENCE_LENGTH,
#                MAX_NB_WORDS,
#                EMBEDDING_DIM,
#                VALIDATION_SPLIT)

# Cluster
knn_method = 'faiss'
k_neighbour = 10
min_size = 20
th_knn = 0.6
th_step = 0.01
max_iter = 100
max_size = 1000
features = np.load(featurefilepath)
faiss_knns = build_faiss_knns(features, knn_dir, knn_method, k_neighbour)
clustgen = generate_cluster(proposal_dir,
                            svertex_predlabelfile,
                            faiss_knns,
                            k_neighbour,
                            th_knn,
                            th_step,
                            max_size,
                            min_size,
                            max_iter)
save_proposals(clustgen, faiss_knns, proposal_dir)

# Evaluate
evaluation(labelfile_path, svertex_predlabelfile)
print('---------------------------\n')

# Kmeans Test
n_clusters = 50
kmeans_feat = kmeans(features, n_clusters)
writepred(kmeans_feat, knn_predlabelfile)
evaluation(labelfile_path, knn_predlabelfile)
print('---------------------------\n')

# Dbscan test
eps = 0.5
min_samples = 100
dbscan_feat = dbscan(features, eps, min_samples)
writepred(dbscan_feat, dbscan_predlabelfile)
evaluation(labelfile_path, dbscan_predlabelfile)

from sklearn.datasets import fetch_20newsgroups

from Baseline.sklearn_cluster import kmeans, dbscan
from Clustering.faiss import build_faiss_knns
from Clustering.gcndcluster import generate_cluster, save_proposals
from Clustering.similarysearch import getnormlize_features
from Evaluation.evaluate import evaluation
from FeatureExtract.featureextract import extractfeature
from FeatureExtract.trainmodel import trainingmodel
from Preprocess.preprocess import preprocess, label2file
from utils import writepred, ifexitdir, np

feature_dir = "FeatureExtract"
knn_dir = "Clustering/knn"
proposal_dir = "Clustering/proposal"
dataset_dir = "Dataset"

# path for offline datasets
# labelfile_path = dataset_dir + "labelfile.txt"
# data_preprocessed_file = "dataset_dir + "raw_data.txt"

labelfile_train = "Preprocess/labelfile_train.txt"
preprocessfile_train = "Preprocess/preprocessfile_train.txt"

labelfile_test = "Preprocess/labelfile_test.txt"
preprocessfile_test = "Preprocess/preprocessfile_test.txt"

modelpath = feature_dir + '/model.bin'
uniquetokenfile = feature_dir + "/unniquetokens.txt"
glovefile = feature_dir + '/glove.6B.100d.txt'

featurefilepath = feature_dir + '/featurefile.npy'

svertex_predlabelfile = "Clustering/proposal/svertex_predlabelfile.txt"
knn_predlabelfile = "Baseline/knn_predlabels.txt"
dbscan_predlabelfile = "Baseline/dbscan_predlabels.txt"

# Reset all generated directory and files for retests
dirarr = [knn_dir, proposal_dir]
filearr = ["featurefilepath"]
ifexitdir(dirarr, filearr)

# Gather dataset and add to memory
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
data_raw = newsgroups_train.data
labels = newsgroups_train.target
classnames = newsgroups_train.target_names

# Preprocess datasets
preprocess(data_raw, preprocessfile_train)
label2file(labels, labelfile_train)

# Train model
classlen = len(classnames)
maxsequence = 1000
max_words = 20000
embed_dim = 100
valid_split = 0.2
trainingmodel(preprocessfile_train,
              labelfile_train,
              classlen,
              glovefile,
              uniquetokenfile,
              modelpath,
              maxsequence,
              max_words,
              embed_dim,
              valid_split)

# get test datasets
newsgroups_train = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
data_raw = newsgroups_train.data
labels = newsgroups_train.target
classnames = newsgroups_train.target_names

# Preprocess datasets
preprocess(data_raw, preprocessfile_test)
label2file(labels, labelfile_test)

# Extract feature
maxsequence = 1000
max_words = 200
extractfeature(preprocessfile_test,
                featurefilepath,
                modelpath,
                maxsequence,
                max_words)

# Cluster
knn_method = 'faiss'
k_neighbour = 10
min_size = 2
th_knn = 0.09 # edges of knns below this value are discarded
th_step = 0.01 # steps to increment to gather all edge conectivity with the edge threshold(th_knn)
max_iter = 100 # iteration of the proposal generation function
max_size = 1000 # max size of super vertices
features = np.load(featurefilepath)
print(features.shape)
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
evaluation(labelfile_test, svertex_predlabelfile)
print('---------------------------\n')

# Kmeans Test
n_clusters = 50
kmeans_feat = kmeans(features, n_clusters)
writepred(kmeans_feat, knn_predlabelfile)
evaluation(labelfile_test, knn_predlabelfile)
print('---------------------------\n')

# Dbscan test
eps = 0.5
min_samples = 100
dbscan_feat = dbscan(features, eps, min_samples)
writepred(dbscan_feat, dbscan_predlabelfile)
evaluation(labelfile_test, dbscan_predlabelfile)

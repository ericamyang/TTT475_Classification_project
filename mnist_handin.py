import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 
from scipy.io import loadmat
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from collections import Counter
import matplotlib.colors as mcolors
import time

class NNClassifier():
    def __init__(self, data_path):
        data_all = loadmat(data_path)
        self.X_train = data_all['trainv']           # training data
        self.y_train = data_all['trainlab'].flatten() # training labels
        self.X_test = data_all['testv']              # test data
        self.y_test = data_all['testlab'].flatten()  # test labels

        self.num_train = data_all['num_train'][0][0]  # number of training samples
        self.num_test  = data_all['num_test'][0][0]   # number of testing samples
        self.row_size  = data_all['row_size'][0][0]    # number of rows in a single image
        self.col_size  = data_all['col_size'][0][0]    # number of columns in a single image
        self.vec_size  = data_all['vec_size'][0][0]    # number of elements in an image vector
        self.num_classes = 10
        self.correct_pred, self.incorrect_pred, self.k_nearest_indices, self.k_nearest_distances = [], [], [], []
        self.k = 7
        print(self.X_train.shape)
    
    def find_error_rates(self, y_pred):
        error_rate = np.sum(y_pred != self.y_test[:len(y_pred)]) / len(self.y_test)
        return error_rate

    def confusion_matrix(self, y_pred):
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        
        for true, pred in zip(self.y_test, y_pred):
            cm[int(true), int(pred)] += 1
        
        return cm

    def nn_classify_chunked(self, target_matrix, labels, chunk_size=1000):
        y_pred = np.zeros(self.num_test, dtype=int)
        self.k_nearest_indices = np.zeros((self.num_test, self.k), dtype=int)
        self.k_nearest_distances = np.zeros((self.num_test, self.k))
        for i in range(0, self.num_test, chunk_size):
            print(f"iteration:{i}")
            end = min(i + chunk_size, self.num_test)
            test_chunk = self.X_test[i:end]
            dists = cdist(test_chunk, target_matrix, metric = 'euclidean')
            nearest_idx = np.argmin(dists, axis = 1)
            k_idxs = np.argsort(dists, axis=1)[:, :self.k]
            self.k_nearest_indices[i:end, :] = k_idxs
            self.k_nearest_distances[i:end, :] = np.take_along_axis(dists, k_idxs, axis=1)
            y_pred[i:end] = labels[nearest_idx] 
            for j in range(0, end-i):
                if y_pred[i+j] == self.y_test[i+j]:
                    self.correct_pred.append(i+j)
                else:
                    self.incorrect_pred.append(i+j)

            print(f"Processed test samples {i} to {end - 1}")
            
        return y_pred    
    
    def cluster_per_class(self, M):
        templates = []  
        template_labels = []  

        for i in range(self.num_classes):
            X_class = self.X_train[self.y_train == i]

            print(f"Clustering class {i} with {len(X_class)} samples...")
            kmeans = KMeans(n_clusters=M, random_state=42)
            kmeans.fit_predict(X_class)

            templates.append(kmeans.cluster_centers_)
            template_labels.extend([i] * M)  

        C_all = np.vstack(templates) 
        labels_all = np.array(template_labels)

        return C_all, labels_all
    
    def knn_classify_chunked(self, target_matrix, labels, chunk_size=1000):
        y_pred = np.zeros(self.num_test, dtype=int)
        self.k_nearest_indices = np.zeros((self.num_test, self.k), dtype=int)
        self.k_nearest_distances = np.zeros((self.num_test, self.k))

        for i in range(0, self.num_test, chunk_size):
            end = min(i + chunk_size,self.num_test)
            test_chunk = self.X_test[i:end]
            dists = cdist(test_chunk, target_matrix, metric = 'euclidean')
            k_idxs = np.argsort(dists, axis=1)[:, :self.k]
            k_labels = labels[k_idxs]
            y_pred[i:end] = np.array([Counter(row).most_common(1)[0][0] for row in k_labels])
            self.k_nearest_indices[i:end, :] = k_idxs
            self.k_nearest_distances[i:end, :] = np.take_along_axis(dists, k_idxs, axis=1)
            for j in range(end - i):
                if y_pred[i+j] == self.y_test[i+j]:
                    self.correct_pred.append(i+j)
                else:
                    self.incorrect_pred.append(i+j)
            print(f"Predicted {i} to {end-1}")
        return y_pred
    
    def plot_misclassified(self, y_pred, num_images=5):
        for i in self.incorrect_pred[:num_images]:
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))

            test_img = self.X_test[i, :].reshape((self.row_size, self.col_size))
            axs[0].imshow(test_img, cmap='gray')
            axs[0].set_title(f"Test idx: {i}\nPred: {y_pred[i]}, True: {self.y_test[i]}")
            axs[0].axis('off')

            neighbor_idx = self.nearest_indices[i]
            train_img = self.X_train[neighbor_idx, :].reshape((self.row_size, self.col_size))
            axs[1].imshow(train_img, cmap='gray')
            axs[1].set_title(f"Template idx: {neighbor_idx}\nLabel: {self.y_train[neighbor_idx]}")
            axs[1].axis('off')

            plt.tight_layout()
            plt.show()

    def plot_classified(self, y_pred, num_images=5):
        for i in self.correct_pred[:num_images]:
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))


            test_img = self.X_test[i, :].reshape((self.row_size, self.col_size))
            axs[0].imshow(test_img, cmap='gray')
            axs[0].set_title(f"Test idx: {i}\nLabel: {self.y_test[i]}")
            axs[0].axis('off')

            neighbor_idx = self.nearest_indices[i]
            train_img = self.X_train[neighbor_idx, :].reshape((self.row_size, self.col_size))
            axs[1].imshow(train_img, cmap='gray')
            axs[1].set_title(f"Template idx: {neighbor_idx}\nLabel: {self.y_train[neighbor_idx]}")
            axs[1].axis('off')

            plt.tight_layout()
            plt.show()



    def plot_misclassified_with_k_nearest(self, y_pred, num_images=5, num_neighbors_to_show=7):
        for i in self.incorrect_pred[:num_images]:
            fig, axs = plt.subplots(1, num_neighbors_to_show + 1, figsize=(3*(num_neighbors_to_show + 1), 4))
            
            test_img = self.X_test[i, :].reshape((self.row_size, self.col_size))
            axs[0].imshow(test_img, cmap='gray')
            axs[0].set_title(f"Test\nPred: {y_pred[i]}\nTrue: {self.y_test[i]}")
            axs[0].axis('off')
            
            neighbor_idxs = self.k_nearest_indices[i, :num_neighbors_to_show]
            for j, idx in enumerate(neighbor_idxs):
                train_img = self.X_train[idx, :].reshape((self.row_size, self.col_size))
                axs[j+1].imshow(train_img, cmap='gray')
                axs[j+1].set_title(f"Neighbor {j+1}\nLabel: {self.y_train[idx]}")
                axs[j+1].axis('off')
            
            plt.tight_layout()
            plt.show()

    def plot_comparison(self, test_idx, compare_indices):
        num_images = 1 + len(compare_indices)
        fig, axs = plt.subplots(1, num_images, figsize=(3.5 * num_images, 4))

        test_vec = self.X_test[test_idx, :]
        test_img = test_vec.reshape((self.row_size, self.col_size))
        axs[0].imshow(test_img, cmap='gray')
        axs[0].set_title(f"Test idx: {test_idx}\nLabel: {self.y_test[test_idx]}")
        axs[0].axis('off')
        
        for i, train_idx in enumerate(compare_indices):
            train_vec = self.X_train[train_idx, :]
            distance = np.linalg.norm(test_vec - train_vec)
            train_img = train_vec.reshape((self.row_size, self.col_size))
            axs[i + 1].imshow(train_img, cmap='gray')
            axs[i + 1].set_title(
                f"Train idx: {train_idx}\nLabel: {self.y_train[train_idx]}\nDist: {distance:.2f}"
            )
            axs[i + 1].axis('off')

        plt.tight_layout()
        plt.show()

def plot_confusion_matrix(cm,err):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], yticklabels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix with error rate:{err}")
    plt.show()

def generate_data_Kmeans():
    np.random.seed(42)
    cluster_1 = np.random.randn(20, 2) * 0.5 + np.array([3, 3])
    cluster_2 = np.random.randn(20, 2) * 0.5 + np.array([-3, 3])
    cluster_3 = np.random.randn(20, 2) * 0.8 + np.array([3, -3])
    cluster_4 = np.random.randn(20, 2) * 0.8 + np.array([-3, -3])
    X = np.vstack((cluster_1, cluster_2, cluster_3, cluster_4))
    return X

def k_means(X, k, max_iters, init_centroids=None):
    np.random.seed(42)
    
    if init_centroids is None:
        centroids = X[np.random.choice(len(X), k, replace=False)]
    else:
        centroids = np.array(init_centroids)
        
    centroid_history = [centroids.copy()]
    assignments_history = []
    distances_history = []

    for iteration in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        assignments = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)
        assignments_history.append(assignments.copy())
        distances_history.append(min_distances.copy())
        
        new_centroids = np.array([X[assignments == ki].mean(axis=0) for ki in range(k)])
        centroids = new_centroids
        centroid_history.append(centroids.copy())
    
    return assignments_history, centroid_history, distances_history

def adjust_color(color, factor):
    rgb = np.array(mcolors.to_rgb(color))
    return rgb * factor + (1 - factor) * np.ones_like(rgb)

def K_means_illustration():
    X = generate_data_Kmeans()
    max_iters = 4

    init_centroids_custom = [(-5,1), (2,5), (1,0), (-2,0.5)]
    assignments_custom, centroids_custom, distances_custom = k_means(X, k=4, max_iters=max_iters, init_centroids=init_centroids_custom)
    assignments_random, centroids_random, distances_random = k_means(X, k=4, max_iters=max_iters)


    all_distances = np.concatenate([np.concatenate(distances_custom), np.concatenate(distances_random)])
    global_min = np.min(all_distances)
    global_max = np.max(all_distances)
    fig, axes = plt.subplots(2, max_iters, figsize=(4 * max_iters, 8))
    color_maps = [
        ['red', 'blue', 'green', 'orange'],
        ['red', 'blue', 'green', 'orange']
    ]

    for row, (assignments_history, centroids_history, distances_history, title) in enumerate([
        (assignments_custom, centroids_custom, distances_custom, "Good centroid initialization"),
        (assignments_random, centroids_random, distances_random, "Bad centroid initialization")
    ]):
        for i in range(max_iters):
            ax = axes[row, i]
            assignments = assignments_history[i]
            centroids = centroids_history[i]
            distances = distances_history[i]
            
            for ki in range(4):
                points = X[assignments == ki]
                point_distances = distances[assignments == ki]
                
                base_color = color_maps[row][ki % len(color_maps[row])]
                
                for point, dist in zip(points, point_distances):
                    norm_dist = (dist - global_min) / (global_max - global_min + 1e-8)
                    adjusted_color = adjust_color(base_color, 0.3 + 0.7 * norm_dist)
                    ax.scatter(point[0], point[1], color=adjusted_color, s=30)
            
            ax.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='x', s=100)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_aspect('equal')
            ax.set_title(f"{title}\nIter {i+1}")

    plt.tight_layout()
    plt.show()

#run function to run KNN on 1-400 clusters per class and plot resulting error rate. Runtime approximatly 1 hour and 30min
def KnnErrorRateVSnumberOfClusters():
    nn_classifier = NNClassifier('data_all.mat')  
    cluster_counts = list(range(1, 401))
    error_rates = []

    for k in cluster_counts:
        C_all, labels_all = nn_classifier.cluster_per_class(k)
        y_pred = nn_classifier.knn_classify_chunked(C_all, labels_all)
        er = nn_classifier.find_error_rates(y_pred)
        error_rates.append(er)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_counts, error_rates, marker='o', markersize=2, linestyle='-')
    plt.xlabel('Clusters per class')
    plt.ylabel('Error Rate')
    plt.title('Error Rate vs Clusters per Class (KNN Classification)')
    plt.grid(True)
    plt.show()

def plot_comparison(test_idx, compare_indices, X_test, X_train, y_test, y_train, row_size, col_size, precomputed_dists=None):
    num_images = 1 + len(compare_indices)
    fig, axs = plt.subplots(1, num_images, figsize=(3.5 * num_images, 4))

    test_vec = X_test[test_idx, :]
    test_img = test_vec.reshape((row_size, col_size))
    axs[0].imshow(test_img, cmap='gray')
    axs[0].set_title(f"Test idx: {test_idx}\nLabel: {y_test[test_idx]}")
    axs[0].axis('off')

    for i, train_idx in enumerate(compare_indices):
        train_vec = X_train[train_idx, :]
        train_img = train_vec.reshape((row_size, col_size))
        if precomputed_dists is not None:
            # Use provided distance if available
            distance = precomputed_dists[i]
        else:
            distance = np.linalg.norm(test_vec - train_vec)
        axs[i + 1].imshow(train_img, cmap='gray')
        axs[i + 1].set_title(
            f"Train idx: {train_idx}\nLabel: {y_train[train_idx]}\nDist: {distance:.2f}"
        )
        axs[i + 1].axis('off')

    plt.tight_layout()
    plt.show()


nn_classifier = NNClassifier('data_all.mat')
nn_classifier_clustered = NNClassifier('data_all.mat')
knn_classifier = NNClassifier('data_all.mat')
knn_classifier_clustered = NNClassifier('data_all.mat')

def task_1a():               
    startTime = time.time()                                          
    y_pred = nn_classifier.nn_classify_chunked(nn_classifier.X_train, nn_classifier.y_train)                #predicted labels from test data
    endTime = time.time()                                                                        
    cm = nn_classifier.confusion_matrix(y_pred)                                                             #setting up the confusion matrix
    error_rate = nn_classifier.find_error_rates(y_pred)                                                     #calculating error rate
    print(f"error_rate NN Without clustering : {error_rate}, with prosessing time {endTime-startTime:2}")     #printing the error 
    plot_confusion_matrix(cm,error_rate)                                                                    #displaying cunfusion matrix

#1.a must be run before this
def task_1b():
    wronglyClassifiedIndxs = [268,241,115]
    for i in wronglyClassifiedIndxs:
        compare_indices = nn_classifier.k_nearest_indices[i]
        compare_dists = nn_classifier.k_nearest_distances[i]
        plot_comparison(i, compare_indices, nn_classifier.X_test, nn_classifier.X_train, nn_classifier.y_test, nn_classifier.y_train, nn_classifier.row_size, nn_classifier.col_size, precomputed_dists=None)

#1.a must be run before this
def task_1c():
    wronglyClassifiedIndxs = [0,5,8]
    for i in wronglyClassifiedIndxs:
        compare_indices = nn_classifier.k_nearest_indices[i]
        compare_dists = nn_classifier.k_nearest_distances[i]
        plot_comparison(i, compare_indices, nn_classifier.X_test, nn_classifier.X_train, nn_classifier.y_test, nn_classifier.y_train, nn_classifier.row_size, nn_classifier.col_size, precomputed_dists=None)


def task_2a():
    global Clusters, labels_all
    startTime = time.time()
    Clusters, labels_all = nn_classifier.cluster_per_class(64)
    endTime = time.time()
    print(f"clustering had a prosessing time of {endTime-startTime}")


def task_2b():
    startTime = time.time()
    y_pred = nn_classifier_clustered.nn_classify_chunked(Clusters, labels_all)                  #predicted labels from test data  
    endTime = time.time()                                                                      
    cm = nn_classifier_clustered.confusion_matrix(y_pred)                                       #setting up the confusion matrix
    error_rate = nn_classifier_clustered.find_error_rates(y_pred)                               #calculating error rate
    print(f"error-rate NN With clustering : {error_rate}, with prosessing time {endTime-startTime}")                                      #printing the error 
    plot_confusion_matrix(cm,error_rate)                                                        #displaying cunfusion matrix

def task_2c():
    startTime = time.time()
    y_pred = knn_classifier_clustered.knn_classify_chunked(Clusters, labels_all) 
    endTime = time.time() 
    cm = knn_classifier_clustered.confusion_matrix(y_pred)                                       #setting up the confusion matrix
    error_rate = knn_classifier_clustered.find_error_rates(y_pred)                               #calculating error rate
    print(f"error-rate KNN With clustering : {error_rate}, with prosessing time {endTime-startTime}")                                      #printing the error 
    plot_confusion_matrix(cm,error_rate)                                                         #displaying cunfusion matrix

def knn_without_clusters():  
    startTime = time.time()                                                       
    y_pred = knn_classifier.knn_classify_chunked(knn_classifier.X_train, knn_classifier.y_train)    #predicted labels from test data  
    endTime = time.time()                                                                       
    cm = knn_classifier.confusion_matrix(y_pred)                                                    #setting up the confusion matrix
    error_rate = knn_classifier.find_error_rates(y_pred)                                             #calculating error rate
    print(f"error-rate NN Without clustering : {error_rate}, with prosessing time {endTime-startTime}")                                       #printing the error 
    plot_confusion_matrix(cm,error_rate)                                                            #displaying cunfusion matrix

def main():
    task_1a()
    task_1b()
    task_1c()
    task_2a()
    task_2b()
    task_2c()
    #K_means_illustration()
    #knn_without_clusters()
    #KnnErrorRateVSnumberOfClusters()

if __name__ == '__main__':
    main()


# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 14:21:11 2021

@author: Ramiro Henriques 52535, Pedro Bailao 53675
"""

import numpy as np
from tp2_aux import images_as_matrix, report_clusters
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt    
from sklearn import cluster

from sklearn.metrics import adjusted_rand_score, silhouette_score

def main():
    
    features = extract_features()

    best_feature, labels_val = get_best_feat(features)

    std, distance, points = calculate_eps(best_feature, labels_val)

  #  distance= distance[distance> 0.50]
   # distance= distance[distance< 1.00]
    #dbscan(std, distance)

    #labelsDB = doDBSCAN(std, 0.95)
    #report_clusters(points, labelsDB, 'DBSCAN.html')
    
    #k_means(std)
    
    #labelsDB = doKMEANS(std, 10)
    #report_clusters(points, labelsDB, 'K-Means.html')
    
    #mean_shift(std)
    
    labelsSM = aux_mean_shift(std, 1.00)
    report_clusters(points, labelsSM, 'Means-Shift.html')
 
   
def extract_features():
    
    try:
        data = np.load('features.npz')
        pca_data = data['pca']
        tsne_data = data['tsne']
        iso_data = data['iso']
        features = np.append(pca_data, np.append(tsne_data, iso_data, axis=1), axis=1)
    except IOError:
        
            image_mat = images_as_matrix()
            pca_data = PCA(n_components=6).fit_transform(image_mat)
            tsne_data = TSNE(n_components=(6), method='exact').fit_transform(image_mat)
            iso_data = Isomap(n_components=6).fit_transform(image_mat)
            features = np.append(pca_data, np.append(tsne_data, iso_data, axis=1), axis=1)
            np.savez('features.npz', pca = pca_data, tsne = tsne_data, iso = iso_data)
    
    return features
    
    
def get_best_feat(features):
    
    label_mat = np.loadtxt('labels.txt', delimiter=',', dtype='int');
    labeled = label_mat[label_mat[:,1]>0,:]
    labels = labeled[:,1]
    ix = labeled[:,0]
    f, prob = f_classif(features[ix,],labels)
    
    best_features = features[:,f>19]
    
    return best_features, labels

def calculate_eps(features, labels):
    
    features = features[:,0:-1:]
    meanFeats = np.mean(features,0);
    stdevFeats = np.std(features,0);
    stdBestFeatures = (features-meanFeats)/stdevFeats;
    
    knc = KNeighborsClassifier(n_neighbors = 5).fit(stdBestFeatures,np.zeros(stdBestFeatures.shape[0]))
    
    distances, _ = knc.kneighbors(return_distance=True)
    
    dist_graph = distances[:,-1]

    np.ndarray.sort(dist_graph)
    
    distance = dist_graph[::-1]
    indexes = np.arange(distance.shape[0])
    
    plt.xlabel('Indexes')
    plt.ylabel('Distances')
    plt.xlim([0,500])
    plt.ylim([0,1])  
    plt.plot(indexes, distance)
    plt.savefig("calculate_eps.png")
    
    return stdBestFeatures, distance, indexes
    
def compute_scores(labels):
    mat = np.loadtxt('labels.txt', delimiter=',', dtype='int');
    labeledIndexes = mat[:,1]>0
    labeledLabels = mat[labeledIndexes]
    comparableLabels = labels[labeledIndexes]
    nSamples = labeledLabels.shape[0]
    
    tp = tn = fp = fn = 0
    
    for i in range(0,nSamples):
        for j in range(i,nSamples):
            positive0 = labeledLabels[i][1] == labeledLabels[j][1]
            positive1 = comparableLabels[i] == comparableLabels[j]
            if positive0 and positive1:
                tp += 1
            elif not positive0 and not positive1:
                tn += 1
            elif positive0 and not positive1:
                fn += 1
            else:
                fp += 1
    
    precision = float(tp)/(fp+tp)
    recall = float(tp)/(fn+tp)
    rand = (tp+tn)/float(tp + tn + fp + fn)
    f1 = 2*((precision*recall)/(precision+recall))
    adjusted_rand = adjusted_rand_score(labeledLabels[:,1],comparableLabels)
    
    return precision, recall, rand, f1, adjusted_rand
    
def plots(parameter_vals, precisions, recalls, rands, f1s, adjusts, silhouettes, graph_name):
    plt.figure()
    plt.title(graph_name)
    plt.plot(parameter_vals, precisions, '-r', label = 'Precision values', color = 'blue') 
    plt.plot(parameter_vals, recalls, '-r', label = 'Recall values', color = 'red')
    plt.plot(parameter_vals, rands, '-r', label = 'Rand values', color = 'green')
    plt.plot(parameter_vals, f1s, '-r', label = 'F1 values', color = 'yellow')
    plt.plot(parameter_vals, adjusts, '-r', label = 'Adjusted rand values', color = 'pink')
    plt.plot(parameter_vals, silhouettes, '-r', label = 'Silhouette values', color = 'purple')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(graph_name)
    
def doDBSCAN(X,eps):
    clust = cluster.DBSCAN(eps=eps)
    labels = clust.fit_predict(X)
    return labels
   
def dbscan(features,dist_graph):
   
    precisions = []
    recalls = []
    rands = []
    f1_scores = []
    adjusted_rands = []
    silhouettes = []
    all_eps = []
    
    for i in range(0, dist_graph.shape[0]):
        labels = doDBSCAN(features,dist_graph[i])
        all_eps.append(dist_graph[i])
        precision, recall, rand, f1, adjusted_rand = compute_scores(labels)
        silhouette = silhouette_score(features,labels)
        precisions.append(precision)
        recalls.append(recall)
        rands.append(rand)
        f1_scores.append(f1)
        adjusted_rands.append(adjusted_rand)
        silhouettes.append(silhouette)
    plots(all_eps, precisions, recalls, rands, f1_scores, adjusted_rands, silhouettes, 'DBSCAN')    

def doKMEANS(X, n):
    centroid, labels, inertia = cluster.k_means(X, n_clusters=n)
    return labels

def k_means(features):
    
    precisions = []
    recalls = []
    rands = []
    f1_scores = []
    adjusted_rands = []
    silhouettes = []
    all_i = []
    for i in range(2,12):
        labels = doKMEANS(features,i)
        all_i.append(i)
        precision, recall, rand, f1, adjusted_rand = compute_scores(labels)
        silhouette = silhouette_score(features,labels)
        precisions.append(precision)
        recalls.append(recall)
        rands.append(rand)
        f1_scores.append(f1)
        adjusted_rands.append(adjusted_rand)
        silhouettes.append(silhouette)
    plots(all_i, precisions, recalls, rands, f1_scores, adjusted_rands, silhouettes, 'KMEANS')
    
def aux_mean_shift(X, n):
    labels = cluster.MeanShift(bandwidth=n).fit_predict(X)
    return labels
def mean_shift(features):
    
    precisions = []
    recalls = []
    rands = []
    f1_scores = []
    adjusted_rands = []
    silhouettes = []
    all_i = []
    
    for i in np.arange(0.8,1.02,0.02):
        labels = aux_mean_shift(features,i)
        all_i.append(i)
        precision, recall, rand, f1, adjusted_rand = compute_scores(labels)
        silhouette = silhouette_score(features,labels)
        precisions.append(precision)
        recalls.append(recall)
        rands.append(rand)
        f1_scores.append(f1)
        adjusted_rands.append(adjusted_rand)
        silhouettes.append(silhouette)
    plots(all_i, precisions, recalls, rands, f1_scores, adjusted_rands, silhouettes, 'MEANS-SHIFT')
              
if __name__ == '__main__':
    main()     
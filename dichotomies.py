import binarytree
from binarytree import models,clustering
from classifip.models import ncc
from scipy import cluster
import numpy as np


class dichotomies:
    
    def __init__(self, arff):
        self.labels = arff.attribute_data['class']
        self.centroids = None
    
    def compute_centroids(self,arff):
        """
        Calculate the (normalized) centroids according to each class value
        """
        for label in self.labels:
            obs = np.array([row[:-1] for row in arff.data if row[-1]==label])
            if not obs.size:
                raise Exception("The following class value has no occurrence in data", label)
            centroid = np.mean(obs,axis=0)
          
            if self.centroids is None :
                self.centroids = centroid
            else:
                self.centroids = np.vstack((self.centroids,centroid))
        
    
    def kmeans_class(self, labels):
        """
        class-based kmeans algorithm for the partition of class values :
        We compute firstly the centroids for each class values, then the kmeans (k=2)
        algorithm is applied to separate the class values into two subsets
        """
        selected_centroids = None
        
        # select the centroids corresponding to the input labels
        for label in labels :
            index = self.labels.index(label)
            if selected_centroids is None :
                selected_centroids = self.centroids[index]
            else :
                selected_centroids = np.vstack((selected_centroids,self.centroids[index]))
        
        selected_centroids = cluster.vq.whiten(selected_centroids)
        codebook,distortion = cluster.vq.kmeans(selected_centroids, 2)

        partition,dist = cluster.vq.vq(selected_centroids, codebook)

        partition = [-1 if i==0 else 1 for i in partition]        
        
        return partition
    
    def dist_centroid(self, arff):
        """
        We compute firstly the centroids for each class values, then we form the
        distance matrix using a specific distance criteria
        """
        self.compute_centroids(arff)
        n = len(self.labels)
        dist = np.zeros((n,n))
        
        for i in range(0,n) :
            for j in range(i+1,n):
                cent_i = self.centroids[i]
                cent_j = self.centroids[j]
                dist[i,j] = np.linalg.norm(cent_i - cent_j)
                
        return dist
        
    def kmeans(self,arff, labels = []):
        """
        Instance-based kmeans algorithm for the partition of class values :
        A kmeans with k=2 is applied to the entry dataset, the representation
        percentage of every class value within each cluster is computed.  
        """
        if labels == [] :
            labels = arff.attribute_data['class']
            freq = [[0,0] for i in arff.attribute_data['class']]
        else :
            freq = [[0,0] for i in labels]
            
        
        obs = []
        obs_label = []
        for row in arff.data :
            if row[-1] in labels :
                obs.append(row[:-1])
                obs_label.append(row[-1])
                
        obs = cluster.vq.whiten(np.matrix(obs))
        codebook,distortion = cluster.vq.kmeans(obs, 2)
        results,dist = cluster.vq.vq(obs, codebook)
        
        for i,res in enumerate(results) :
            index = labels.index(obs_label[i])
            freq[index][res] += 1
             
        sum1 = sum(results)
        sum0 = len(results) - sum1

        freq = [[float(f[0])/sum0,float(f[1])/sum1] for f in freq]
        
        partition = []

        for f in freq:
            if f[0] > f[1]:
                partition.append(0)
            else : 
                partition.append(1)
        return partition
    
    
    def tree_kmeans(self,arff,labels=[], tree_parent = None):
        """
        Construct the dichotomy tree using kmeans algorithm. 
        Attention, a specific strategy must be chosen : either "class-based" or
        "instance-based"
        """
        labels_left = []
        labels_right = []        
        
        # Initialization of the recursion: 
        if labels == []:
            self.compute_centroids(arff)
            tree_p = models.BinaryTree(label=self.labels)
            self.tree_kmeans(arff, labels=self.labels, tree_parent = tree_p)
            return tree_p
        
        # Recursion :       
        if len(labels) > 1 :
            if len(labels) == 2 :
                tree_parent.left = models.BinaryTree(label=[labels[0]])
                tree_parent.right = models.BinaryTree(label=[labels[1]])
            else:
#                 partition = self.kmeans_class(labels)
                partition = self.kmeans(arff,labels)
        
                for index, class_val in enumerate(labels):
                    if partition[index] > 0 :
                        labels_left.append(class_val)
                    else : 
                        labels_right.append(class_val)
        
                if (labels_left == []) or (labels_right == []) :
                    raise Exception("Error when splitting labels",labels,partition)
                
                tree_parent.left = models.BinaryTree(label=labels_left)
                tree_parent.right = models.BinaryTree(label=labels_right)
                
                self.tree_kmeans(arff, labels_left, tree_parent.left)
                self.tree_kmeans(arff, labels_right, tree_parent.right)
            
    def build_ordinal(self,arff,labels=[], tree_parent=None):
        """
        Building the dichotomy tree using the ordinal structure of the class.
        At each split, every possibility in respect of the ordinal structure is tested, 
        the one yielding the best accuracy with the training dataset is chosen. 
        """
        #Initialization
        if labels == []:
            tree_p = models.BinaryTree(label=self.labels)
            self.build_ordinal(arff, self.labels, tree_p)
            return tree_p
        
        #Recursion
        elif len(labels) > 1 :
            # A NBC classifier is learnt for each split, the accuracy of each classifier is stored
            scores = [0 for l in range(len(labels)-1)]
            for index,label in enumerate(labels):
                nbc = ncc.NCC()
                # We split labels using the ordinal information
                if index > 0 :
                   
                    data_bi = binarytree.select_class_binary(arff,positive=labels[0:index], negative=labels[index:])
                
                    nbc.learn(data_bi)               
                    evaluations = nbc.evaluate(data_bi.data, ncc_s_param=[2])
                    results = []
                    
                    for eva in evaluations: 
                        results.append(eva[0].nc_maximal_decision())
                        
                    data_lab=[x[-1] for x in data_bi.data]
                    
                    for run,lab in enumerate(data_lab):
                        ind = nbc.feature_values['class'].index(lab)
                        if results[run][ind] == 1:
                            sc = 1./results[run].sum()
                            scores[index-1] += - 1.2* sc*sc + 2.2 * sc
                  
            
            # The NBC yielding the highest score is chosen as the actual split
            ind_max = scores.index(max(scores)) + 1
            labels_left = labels[0:ind_max]
            labels_right = labels[ind_max:]
            
            
            tree_parent.left = models.BinaryTree(label=labels_left)
            tree_parent.right = models.BinaryTree(label=labels_right)
            
            self.build_ordinal(arff, labels_left, tree_parent.left)
            self.build_ordinal(arff, labels_right, tree_parent.right)
                
    
    def build_hierarchical(self,arff):
        """
        Building the dichotomy tree with the hirarchical clustering method. Basing on the distance matrix, 
        a set of linkage techniques are used. The one yielding the best accuracy over training data is retained. 
        """
        clusters = clustering.clustering(arff)
        clusters.compute_distances()
        
        #Initialization 
        linkages = {"single":None,"complete":None,"average":None,"median":None,"centroid":None,"weighted":None,"ward":None}
        discounted_acc = [0. for i in range(0,7)]
        nb_var = len(clusters.feature_names)
        data_validation = []
        lab_validation = []
        for instance in arff.data:
            data_validation.append(instance[:(nb_var-1)])
            lab_validation.append(instance[-1])
        
        for index, method in enumerate(linkages.keys()):
            tree = clusters.build_tree(method)
            tree.learnAll(arff)
            linkages[method]=tree
            
            tree.evaluate(data_validation,ncc_s_param=[2])    
            results = tree.decision_maximality()
                
                
            for run,lab in enumerate(lab_validation):
                ind = clusters.feature_values['class'].index(lab)
                if results[run][ind]==1.:
                    score = 1/results[run].sum()
                    discounted_acc[index] += -1.2 * score * score + 2.2 * score
            
        index_max = discounted_acc.index(max(discounted_acc))
            
        return linkages.values()[index_max]
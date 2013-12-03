
from binarytree import models
import numpy as np
import math
from scipy import cluster

class clustering:
    def __init__(self,arff):
        
        self.labels = arff.attribute_data['class']
        n = len(self.labels)
        self.distances = np.zeros((n,n))
        
        #Initializing the counts
        self.label_count = []
        self.feature_count=dict()
        self.feature_names=arff.attributes[:]
        self.feature_values=arff.attribute_data.copy()
        for class_value in arff.attribute_data['class']:
            subset=[row for row in arff.data if row[-1]==class_value]
            self.label_count.append(len(subset))
            for feature in arff.attributes[:-1]:
                count_vector=[]
                feature_index=arff.attributes.index(feature)
                for feature_value in arff.attribute_data[feature]:                   
                    nb_items=[row[feature_index] for row in subset].count(feature_value)
                    count_vector.append(nb_items)
                self.feature_count[class_value+'|'+feature]=count_vector
            
            
    def compute_distances(self):
        
        for y1 in self.labels :
            indice1 = self.labels.index(y1)
            
            for y2 in self.labels[indice1:] :
                indice2 = self.labels.index(y2)
                dist = 0
                
                for x in self.feature_names[:-1] :
                    count1 = [float(coef)/self.label_count[indice1] for coef in self.feature_count[y1 + '|' + x]]
                    count2 = [float(coef)/self.label_count[indice2] for coef in self.feature_count[y2 + '|' + x]]
                    
                    #Mesure de proximite :                    
                    coef = 0
                    # Jensen-Shannon    
#                     for i in range(0,len(count1)-1) :
#                         if count1[i] == 0:
#                             if count2[i] > 0 :
#                                 coef += count2[i]/2
#                         else :
#                             if count2[i] == 0 :
#                                 coef += count1[i]/2
#                             else:
#                                 prob = (count1[i] + count2[i])/2
#                                 coef += count1[i] * math.log(count1[i]/prob,2)/2 + count2[i] * math.log(count2[i]/prob,2)/2
#                              
#                     dist += math.sqrt(coef)
                    
                    # Bhattacharyya 
                    for i in range(0,len(count1)-1) :
                        coef += math.sqrt(count1[i]*count2[i])
                             
                    dist = dist + math.sqrt(math.fabs(1 - coef))
                      
                self.distances[indice1,indice2] = dist/(len(self.feature_names[:-1]))
                
    def build_tree(self,method='single'):
        clusters = cluster.hierarchy.linkage(self.distances,method=method)
        trees = []
        length = len(self.labels)
        i = length
        for link in clusters :
            i = i + 1
            if (int(link[0]) < length) :
                l_label = [self.labels[int(link[0])]]
                l_tree = models.BinaryTree(label=l_label) 
            else :
                l_tree = trees[int(link[0])-length]
                l_label = l_tree.node.label
                
            if (int(link[1]) < length) :
                r_label = [self.labels[int(link[1])]]
                r_tree = models.BinaryTree(label=r_label)
            else :
                r_tree = trees[int(link[1])-length]
                r_label = r_tree.node.label
                    
            p_label = l_label + r_label         
            parent = models.BinaryTree(label=p_label)
            parent.left = l_tree
            parent.right = r_tree
            trees.append(parent)
            
        return trees[-1]

            
            
            
            
            
'''
Created on 21 avr. 2014

@author: Gen
'''
from binarytree import models
from scipy.stats import rv_discrete
from classifip.representations import intervalsProbability as IP
import numpy as np
import random
import math

class Forest:
    '''
    A list of 'BinaryTree' type structures
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.listTree = []
        self.length = 0
        self.class_values = None
        self.size_data = None
    
                
    def genTree(self,labels):
        '''
        Generate a single random tree structure uniformly among all potential trees 
        '''
        n = len(labels) - 1
        bitcodes = ''
        #Generate first a chain of bits of size 2n+1
        i = 0 
        j = 0
        while i+j < 2*n + 1:
            if i == n : #if n '0' are already generated, we complete by adding '1'
                bitcodes = bitcodes + '1'
                j+=1
            elif j == n+1: #if n '1' are already generated, we complete by adding '0'
                bitcodes = bitcodes + '0'
                i+=1 
            else : #generate the next bit
                proba = float(n-i)/(2*n+1-i-j) #proba of having '0' generated
                distrib = rv_discrete(values=((0,1),(proba,1-proba)))
                
                #generate a bit according to the defined proba distribution
                newbit = rv_discrete.rvs(distrib) 
                bitcodes = bitcodes + str(newbit)
                
                #increment i or j according to the generated bit
                if newbit == 0 :
                    i+=1
                else:
                    j+=1
                
        
        # find the minimum partial sum knowing '0' weights 1 and '1' weights -1
        psum = 0
        mini = 1
        index_min = 0
        for k in range(0,2*n+1):
            psum += 1 if bitcodes[k] == '0' else -1
            if psum < mini :
                mini = psum
                index_min = k
        
        if index_min < 2*n+1:
            #form a new bitcodes starting by the element n_(index_min+1) of the old one
            new_bitcodes = bitcodes[index_min+1:] + bitcodes[0:index_min+1]
            #we do the same permutation with the labels
            #----------------- index_labels = bitcodes[0:index_min+1].count('1')
            #------- new_labels = labels[index_labels:] + labels[0:index_labels]
        else:
            #if the minimum is reached with the whole set, there is no permutation
            new_bitcodes = bitcodes

        #Randomize/shuffle the vector of labels
        new_labels = labels
        random.shuffle(new_labels)
    
            
        
        #Find where ends the first (and minimal) regular prefix bit codes
        def regular(codes):
            psum=0
            length=0
            while psum <> -1: #a regulet prefix code is weighted to -1
                if length >= len(codes) : raise Exception('Bad encoding',codes)
                if codes[length] =='0':
                    psum += 1
                else :
                    psum -= 1
                length+=1
            
            # return the length of the first regular prefix code 
            return length
                
        
        #the newly generated bitcodes can be transformed into a tree (Lukasiewicz coding)
        def build(codes,labels_node): #this function transform recursively a Lukasiewicz code into tree

            if codes == '1': #the case where there is only label
                tree = models.BinaryTree(label=labels_node)
                return tree
            else :
                if codes[0] <> '0':
                    raise Exception('Bad tree-coding bit codes', codes)
                else:
                    tree = models.BinaryTree(label=labels_node)
                    
                    # build the left child-node
                    length_left = regular(codes[1:]) #the length of the bitcodes of the left child
                    codes_left = codes[1:1+length_left] #bitcodes of the left child
                    tree.left = build(codes_left,labels_node[0:codes_left.count('1')])
                    
                    
                    # build the right child-node
                    codes_right = codes[1+length_left:]
                    tree.right = build(codes_right,labels_node[codes_left.count('1'):])
                    
                    return tree

        return build(new_bitcodes, new_labels)
    
    
    def genOrdinalTree(self,labels):
        '''
        Generate an random ordinal tree given the set of initial labels.
        At every node, we with randomly the set of labels in two subsets, until
        all leaf-nodes are singletons
        '''
        tree = models.BinaryTree(label=labels)
        nb_labels = len(labels)
        
        if (nb_labels > 1):
            split = random.randint(1,nb_labels-1)
            tree.left = self.genOrdinalTree(labels=labels[0:split])
            tree.right = self.genOrdinalTree(labels=labels[split:])
        else :
            tree.left = None
            tree.right = None
        
        return tree        
        
        
    def genForest(self,dataArff,n=10,struct='multiclass',random_seed=None):
        '''
        Generate uniformly a forest of tree structures using the genTree function
        There is no control of redundancy or symmetry
        <=> We assume that these effects would cancel between themselves 
        
        :param struct: state if the class has a specific structure, maybe either
        "multiclass" or "ordinal"
        '''
        labels = dataArff.attribute_data['class']
        self.length = n
        
        if struct == 'multiclass':
            for i in range(0,n):
                self.listTree.append(self.genTree(labels))
        elif struct == 'ordinal':
            if random_seed is not None:
                random.seed(random_seed)
            for i in range(0,n):
                self.listTree.append(self.genOrdinalTree(labels))
        else:
            raise Exception('Unknown class structure : ',struct)
        
    def learnForest(self,dataArff):
        '''
        Learn prior probabilities using training data for each tree
        '''
        for i in range(0,self.length):
            self.listTree[i].learnAll(dataArff)
    
    def evaluateForest(self,dataArff,ncc_s_param=[2]):
        '''
        Evaluate a validation data set for each tree, using NCC : conditional
        posterior probabilities for each node of trees are computed and stocked
        
        '''
        # preparing the data set for evaluation
        self.class_values = dataArff.attribute_data['class']
        val_data = [x[0:len(x)-1] for x in dataArff.data]
        self.size_data = len(val_data)
        
        #Internal function taking 2 Intervals Probability and restraining them 
        #=======================================================================
        # def majProba(ip1,ip2):
        #     lproba1 = ip1.lproba
        #     lproba2 = ip2.lproba
        #     return np.array([[np.min([lproba1[0],lproba2[0]], 0)],[np.max([lproba1[1],lproba2[1]], 0)]])
        #=======================================================================
            
            
        #Firstly we find p_u/p_l for each tree
        for i in range(0,self.length): 
            self.listTree[i].evaluate(val_data,ncc_s_param=ncc_s_param)
            
    
    def best_tree(self,arff,random_seed=None,ncc_s=[1],costs=None):
        '''
        Return the dichotomy tree with the best accuracy u65
        '''
        
        size_data = len(arff.data)
        discounted_acc = [0.] * self.length
        
        if random_seed != None:
            random.seed(random_seed)
            random.shuffle(arff.data)
            
        datatr=arff.make_clone()            
        datatr.data = arff.data[0:(3*size_data)/4]
                    
        data_validation = []
        lab_validation = []
        for instance in arff.data[(3*size_data)/4:]:
            data_validation.append(instance[:-1])
            lab_validation.append(instance[-1])
        
        for index, tree in enumerate(self.listTree):
            tree.learnAll(datatr)
            
            tree.evaluate(data_validation,ncc_s_param=ncc_s)    
            results = tree.decision_maximality(costs=costs)
                
            for run,lab in enumerate(lab_validation):
                    ind=tree.node.label.index(lab)
                    if results[run][ind]==1.:
                        score = 1/results[run].sum()
                        discounted_acc[index] += -0.6 * score * score + 1.6 * score
            
        index_max = discounted_acc.index(max(discounted_acc))
        
        return self.listTree[index_max]
                
                
    def decision_maximality(self,costs=None,decision='mean'):
        
        nbData = self.size_data
        class_values = self.class_values
        maximality_class=np.ones((nbData,len(class_values)))
         
            
        nb_class = len(class_values)
        
        if costs is None :
            costs = 1 - np.eye(nb_class)

      
        def lowerExp(x,y,cost_x,cost_y):
            expInf = np.zeros(nbData) 
            if 'numpy.ndarray' in str(type(cost_x)) and 'numpy.ndarray' in str(type(cost_y)):
                for k in range(0,nbData):                
                    expInf[k] = min(cost_x[k]*x.node.proba[k][0]+cost_y[k]*y.node.proba[k][1],cost_x[k]*x.node.proba[k][1]+cost_y[k]*y.node.proba[k][0])
            elif 'numpy.ndarray' in str(type(cost_x)) :
                for k in range(0,nbData):                 
                    expInf[k] = min(cost_x[k]*x.node.proba[k][0]+cost_y*y.node.proba[k][1],cost_x[k]*x.node.proba[k][1]+cost_y*y.node.proba[k][0])
            elif 'numpy.ndarray' in str(type(cost_y)) :
                for k in range(0,nbData):                 
                    expInf[k] = min(cost_x*x.node.proba[k][0]+cost_y[k]*y.node.proba[k][1],cost_x*x.node.proba[k][1]+cost_y[k]*y.node.proba[k][0])
            else :
                for k in range(0,nbData):                 
                    expInf[k] = min(cost_x*x.node.proba[k][0]+cost_y*y.node.proba[k][1],cost_x*x.node.proba[k][1]+cost_y*y.node.proba[k][0])
            return expInf
            
        def recc(cost,x,y):
            
            if x.node.count() == 1 and y.node.count() == 1 :
                return lowerExp(x,y,cost[class_values.index(x.node.label[0])],cost[class_values.index(y.node.label[0])])
            
            elif (x.node.count() == 1) or (y.node.count() == 1) :
                if x.node.count() == 1 : #'x' is singleton
                    return lowerExp(x,y,cost[class_values.index(x.node.label[0])],recc(cost,y.left,y.right))
                elif y.node.count() == 1 :
                    return lowerExp(x,y,recc(cost,x.left,x.right),cost[class_values.index(y.node.label[0])])
                else :
                    raise Exception('Unexpected error')
            else :
                return lowerExp(x,y,recc(cost,x.left,x.right),recc(cost,y.left,y.right))
            
        if decision == 'mean' :    
            for i in range(0,nb_class):
                for j in range(i+1,nb_class):
                    cost_i_j = costs[i] - costs[j]
                    expInf_ij = 0.
                    expInf_ji = 0.
                    
                    # Compute the sum of expInf over all trees 
                    for k in range(0,self.length):
                        #Verify if i is dominated by j
                        expInf_ij += recc(cost_i_j,self.listTree[k].left,self.listTree[k].right)
                        
                        #Verify if j is dominated by i
                        expInf_ji += recc(-cost_i_j,self.listTree[k].left,self.listTree[k].right)
                        
                    # The sign of the sum of expInf determine the dominance
                    for ind in range(0,self.size_data):
                        if expInf_ij[ind] > 0 :
                            maximality_class[ind,i] = 0
                        if expInf_ji[ind] > 0 :
                            maximality_class[ind,j] = 0
        elif decision == 'median':
            for i in range(0,nb_class):
                for j in range(i+1,nb_class):
                    cost_i_j = costs[i] - costs[j]
                    expInf_ij = []
                    expInf_ji = []
                    
                    # Compute the expInf for each trees, then order them 
                    for k in range(0,self.length):
                        #Verify if i is dominated by j
                        expInf_ij.append(recc(cost_i_j,self.listTree[k].left,self.listTree[k].right))
                        
                        #Verify if j is dominated by i
                        expInf_ji.append(recc(-cost_i_j,self.listTree[k].left,self.listTree[k].right))
                    
                    expInf_ij = np.sort(np.array(expInf_ij).reshape((self.size_data,self.length)))
                    expInf_ji = np.sort(np.array(expInf_ji).reshape((self.size_data,self.length)))
                    
                    
                    # The sign of the median among the vector of expInf determine the dominance
                    if self.length % 2 == 0 :
                        mediane = int(math.floor(self.length*2./3.))
                    else :
                        mediane = int(math.floor(self.length*2./3.)+1)

                    for ind in range(0,self.size_data):
                        if expInf_ij[ind][mediane] > 0 :
                            maximality_class[ind,i] = 0
                        if expInf_ji[ind][mediane] > 0 :
                            maximality_class[ind,j] = 0
        elif decision == 'vote' :
            for tree in self.listTree:
                #Perform a majority vote where a voter can split one vote to several classes
                decisions = tree.decision_maximality()
                #===============================================================
                # for i in range(0,self.size_data):
                #     decisions[i] = decisions[i] /np.sum(decisions[i])
                #===============================================================
                maximality_class += decisions
            for i,row in enumerate(maximality_class):
                for j,col in enumerate(row):
                    if col <= self.length/2:
                        maximality_class[i,j] = 0
                    else:
                        maximality_class[i,j] = 1.
        else :
            raise Exception('Unspecified decision criteria')
        
                        
        return maximality_class
    
    def decision_intervaldom(self,costs=None,decision='vote'):
        nbData = self.size_data
        class_values = self.class_values
        predictions = np.ones((nbData,len(class_values)))
         
            
        nb_class = len(class_values)
        
        if decision=='vote':
            for tree in self.listTree:
                #Perform a majority vote where a voter can split one vote to several classes
                decisions = tree.decision_intervaldom()
                #===============================================================
                # for i in range(0,self.size_data):
                #     decisions[i] = decisions[i] /np.sum(decisions[i])
                #===============================================================
                predictions += decisions
            for i,row in enumerate(predictions):
                for j,col in enumerate(row):
                    if col <= self.length/2:
                        predictions[i,j] = 0
                    else:
                        predictions[i,j] = 1.
        elif decision == 'mean' :
            probas = [0] * nbData
            for tree in self.listTree:
                for index, ip in enumerate(tree.probaPos()):
                    probas[index] += ip.lproba
            
            
            for i in range(0,nbData):
                ip_final = IP.IntervalsProbability(probas[i]/self.length)
                predictions[i] = ip_final.nc_intervaldom_decision()
                
        else:
            raise Exception('Unspecified decision criteria') 
        
        return predictions
    
    def probaPos(self,bound=0):
        """
        Calculate the posterior probabilities
         
        :param bound: set to '0' to get lower bound and '1' for upper bound and
        '[0,1]' for both
        :tyope bound: either int or [int,int]
        
        Attention, currently it's not possible to get both bounds
        """
         
        ret = np.zeros((self.size_data,len(self.class_values)))
         
        for i in range(0,self.length):
            proba_list = self.listTree[i].probaPos(bound=bound)

            for k,proba in enumerate(proba_list):
                ret[k] += proba

        return ret/self.length
            
'''
Created on 10 avr. 2013

@author: Gen YANG
'''
import random
import numpy as np
from binarytree import select_class_binary
from classifip.models import ncc


class Node:
    """
    Node of an binary tree. Attributes are :
        
    :param label: list of class values associated with the node
    :type label: lsit of strings
    
    :param proba: the probability associated with the class values according to
    the binary classification problem
    :type proba: ???
    """

    
    def __init__(self, label=None):
        """
        Constructor : create a node and associate class values to it.
        """
        self.label = label
        self.proba = []
        self.learner = ncc.NCC()
        
        
    def __str__(self):
        """
        Method for printing a node
        """
        s = str(self.label) + " \n"
                
        return s
        
        
    def isEmpty(self):
        """
        Testing if the node is "empty"
        :rtype: boolean
        """
        return (self.label is None)
    
    
    def count(self):
        """
        Count the number of labels/class values. 
        :rtype: integer
        """
        return len(self.label)    
            
            
    def splitNode(self, method = "random", occurrences=None, label = None):
        """
        Split a node into two sub-nodes
        :param method: diffrent method for splitting a set of class values
        
        :param label: if this argument is not None, then its content are used to
                    form the 'left child' of the current node, and the rest of 
                    current node's class values form the 'right child node'.
        :type label: list of string (must be class values)
        
        :rtype: Node, Node
        """
        if self.isEmpty() :
            raise Exception("Cannot split an empty node")

        if label is not None :
            '''
            For manual splitting only 
            '''
            left = Node(label)
            right = Node(list(set(self.label) - set(left.label))) 
        
        elif method == "random" :
            '''
            Split randomly class values in two balanced subsets
            '''
            left = Node(random.sample(self.label, len(self.label)/2))
            right = Node(list(set(self.label) - set(left.label)))  
            
        elif method == "data-balanced" :
            '''
            Split class values in two subsets so that the number of instances is 
            'approximatively equal' for both subsets.
            '''
            keys = sorted(occurrences, key=occurrences.get)
            total = sum(occurrences.values())
            l_label = [keys[-1]]
            i = 0
            buff = occurrences[keys[-1]]
            while buff < total/2 :
                buff += occurrences[keys[i]]
                l_label.append(keys[i])
                i += 1
            
            left = Node(l_label)
            right = Node(list(set(self.label) - set(l_label)))
            
        else :
            raise Exception("Unrecognized splitting method")    
            
        return left,right
    
       


class BinaryTree:
    """
    Binary tree representing the structure of the nested dichomoties of a 
    multilabel classification problem. 
    
    Each node represents an ensemble of class values ans its probability 
    according to the results of the binary classification.
    
    :param node: the root node 
    :type node: Node
    
    :param left: the left child subtree
    :type left: BinaryTree
          
    :param right: the right child subtree
    :type right: BinaryTree
    """


    def __init__(self, node=None, label=None):
        """
        Constructor : initialize the root node and its children tree. 
            - node : current node
            - left : left child subtree 
            - right : right child subtree 
        """
        if node is None:
            self.node = Node(label=label)
        else :    
            self.node = node
             
        self.left = None
        self.right = None
        
     
    def build(self, method="random",occurrences=None):
        """
        Build the structure of the entire binary tree by splitting the initial
        root node (the ensemble of class values) into children nodes.
        """
        if self.node.isEmpty() :
            raise Exception("Cannot split an empty root node")
        
        if (self.left is not None) or (self.right is not None) :
            raise Exception("The given root node already has child")
        
        if self.node.count() == 1 :
            '''
            When there is only one class value in the label of the current node,
            we stop splitting.
            '''
            self.left = None
            self.right = None
        
        else :
            if method == "data-balanced":
                if occurrences is None:
                    raise Exception('A dictionary of class values occurrences is required. Use count_labels function to get the dictionary')
             
            l_node, r_node = self.node.splitNode(method,occurrences)
            self.left = BinaryTree(node = l_node)
            self.right = BinaryTree(node = r_node)
            
            self.left.build(method,{key:occurrences[key] for key in occurrences 
                                    if key not in r_node.label})
            self.right.build(method,{key:occurrences[key] for key in occurrences 
                                     if key not in l_node.label})     
            

    
    def learnCurrent(self,dataset):
        """
        Learn the underlying binary classification problem associated with the 
        current node. Actions performed :
            - Render the incoming data (for the global multibal classification) 
            accordingly : select the useful datalines, remember selected indices
            - discretize continue variables (to-do)
        :return: the indices of selected data in the original dataset
        :rtype: list of integers
        """
        
        if self.left.node.isEmpty() or self.right.node.isEmpty() :
            raise Exception("Current node has no left or/and right child node.")
        
        data = select_class_binary(dataset,positive=self.left.node.label, 
                                       negative=self.right.node.label)
        
        # learning process for the current node
        self.node.learner = ncc.NCC()
        self.node.learner.learn(data)


    
    def learnAll(self,dataset):
        """
        Iterate the learning process (cf. 'learnCurrent') for the entire structure
        """
        if (self.left is not None) and (self.right is not None) :
            self.learnCurrent(dataset)
            '''
            we only try to learn the children nodes when there are more than one
            class value / label associated with them.
            '''
            if self.left.node.count() > 1:
                self.left.learnAll(dataset)
            if self.right.node.count() > 1:    
                self.right.learnAll(dataset)      
        
    
    def evalAll(self,testdataset,ncc_epsilon,ncc_s_param): 
        """
        Recursive evaluation process for the entire structure
        """            
        if self is not None :
            # for a single node
            self.left.node.proba = [] #we reset results of previous evaluations
            self.right.node.proba = []
            result = self.node.learner.evaluate(testdataset, ncc_epsilon, ncc_s_param)
            for ip in result:
                self.left.node.proba.append([ip[0].lproba[1,0],ip[0].lproba[0,0]])
                self.right.node.proba.append([ip[0].lproba[1,1],ip[0].lproba[0,1]])
            
            # recursion
            if self.left.node.count() > 1:
                self.left.evalAll(testdataset,ncc_epsilon,ncc_s_param)
            if self.right.node.count() > 1:
                self.right.evalAll(testdataset,ncc_epsilon,ncc_s_param)

    
    def evaluate(self,testdataset,ncc_epsilon=0.001,ncc_s_param=[2]):
        """
        evaluate the instances and set the probability intervals for each node.
        
        :param testdataset: list of input features of instances to evaluate
        :type dataset: list
        :param ncc_epsilon: espilon issued from [#corani2010]_ (should be > 0)
            to avoid zero count issues
        :type ncc_espilon: float
        :param ncc_s_param: s parameters used in the IDM learning (settle
        imprecision level)
        :type ncc_s_param: list
        """

        if self is not None :
            self.node.proba = np.ones((len(testdataset),2))
            self.evalAll(testdataset, ncc_epsilon, ncc_s_param)
            
            
    
    def decision_maximality(self):
        """
        Return the classification decisions using using diffrent deicision criteria:
            - 'intervaldom' : interval dominance
        
        :return: the set of optimal classes (under int. dom.) as a 1xn vector
            where indices of optimal classes are set to one
        :rtype: :class:`~numpy.array`
        """
        
        if self.left is None or self.right is None :
            raise Exception('No child')
        
        nbData = len(self.node.proba)
        class_values = self.node.label
        maximality_class=np.ones((nbData,len(class_values)))
        
        #internal method for comparing two Intervals Probabilities
        def compare(i,x,y,x_proba,y_proba):
            return (x.node.proba[i][0]*x_proba[i,0] >
                    y.node.proba[i][1]*y_proba[i,1])
        
        def maximality_loop(x,y,x_proba,y_proba):
            """
            x_proba and y_proba are used here to stock and accumulate the IP of 
            the parents nodes shared by currently evaluated nodes 'x' and 'y'.
            They are represented as a list (depending on the size of testdata) of
            [lower bound, upper bound] (P(x|parents nodes of x))
            
            """

            #if both left and right children are singletons
            if (x.node.count() == 1) and (y.node.count() == 1) :
                #Verify the domination relation between x and y
                
                for i in range(0,nbData):
                    if compare(i,x,y,x_proba,y_proba): #y is dominated
                        maximality_class[i,class_values.index(y.node.label[0])] = 0
                    elif compare(i,y,x,y_proba,x_proba): #x is dominated
                        maximality_class[i,class_values.index(x.node.label[0])] = 0 
            #if only one child is singleton
            elif (x.node.count() == 1) or (y.node.count() == 1) :
                if x.node.count() == 1 : #'x' is singleton
                    maximality_loop(y.left,y.right,y_proba.copy(),y_proba.copy())
                    
                    for i in range(0,nbData): #accumulate the ip of 'y'
                        y_proba[i,0] *= y.node.proba[i][0]
                        y_proba[i,1] *= y.node.proba[i][1] 
                    
                    maximality_loop(x,y.left,x_proba.copy(),y_proba.copy())
                    maximality_loop(x,y.right,x_proba.copy(),y_proba.copy())
                    
                elif y.node.count() == 1 :
                    maximality_loop(x.left,x.right,x_proba.copy(),x_proba.copy())
                    
                    for i in range(0,nbData):
                        x_proba[i,0] *= x.node.proba[i][0]
                        x_proba[i,1] *= x.node.proba[i][1]
                    
                    maximality_loop(x.left,y,x_proba.copy(),y_proba.copy())
                    maximality_loop(x.right,y,x_proba.copy(),y_proba.copy())
                    
                else :
                    raise Exception('Unexpected error')
                    
            else: #both children are not singletons
                maximality_loop(x.left,x.right,x_proba.copy(),x_proba.copy())
                maximality_loop(y.left,y.right,y_proba.copy(),y_proba.copy())
                
                for i in range(0,nbData):
                    x_proba[i,0] *= x.node.proba[i][0]
                    x_proba[i,1] *= x.node.proba[i][1] 
                    y_proba[i,0] *= y.node.proba[i][0]
                    y_proba[i,1] *= y.node.proba[i][1]
                      
                    
                maximality_loop(x,y.left,x_proba.copy(),y_proba.copy())
                maximality_loop(x,y.right,x_proba.copy(),y_proba.copy())
                maximality_loop(x.left,y,x_proba.copy(),y_proba.copy())
                maximality_loop(x.right,y,x_proba.copy(),y_proba.copy())

                 
        maximality_loop(self.left,self.right,self.node.proba.copy(),self.node.proba.copy())
        return maximality_class
    
    def decision_intervaldom(self):
        """
        Return the classification decisions using using diffrent deicision criteria:
            - 'intervaldom' : interval dominance
        
        :return: the set of optimal classes (under int. dom.) as a 1xn vector
            where indices of optimal classes are set to one
        :rtype: :class:`~numpy.array`
        """
        
        if self.left is None or self.right is None :
            raise Exception('No child')
        
        nbData = len(self.node.proba)
        class_values = self.node.label
        maximality_class=np.ones((nbData,len(class_values)))
        
        #internal method for comparing two Intervals Probabilities
        def compare(i,x,y,x_proba,y_proba):
            return (x.node.proba[i][0]*x_proba[i,0] >
                    y.node.proba[i][1]*y_proba[i,1])
        
        def maximality_loop(x,y,x_proba,y_proba):
            """
            x_proba and y_proba are used here to stock and accumulate the IP of 
            the parents nodes shared by currently evaluated nodes 'x' and 'y'.
            They are represented as a list (depending on the size of testdata) of
            [lower bound, upper bound] (P(x|parents nodes of x))
            
            """

            #if both left and right children are singletons
            if (x.node.count() == 1) and (y.node.count() == 1) :
                #Verify the domination relation between x and y
                
                for i in range(0,nbData):
                    if compare(i,x,y,x_proba,y_proba): #y is dominated
                        maximality_class[i,class_values.index(y.node.label[0])] = 0
                    elif compare(i,y,x,y_proba,x_proba): #x is dominated
                        maximality_class[i,class_values.index(x.node.label[0])] = 0 
            #if only one child is singleton
            elif (x.node.count() == 1) or (y.node.count() == 1) :
                if x.node.count() == 1 : #'x' is singleton
                    for i in range(0,nbData): #accumulate the ip of 'y'
                        y_proba[i,0] *= y.node.proba[i][0]
                        y_proba[i,1] *= y.node.proba[i][1] 
                    
                    maximality_loop(y.left,y.right,y_proba,y_proba.copy())
                    maximality_loop(x,y.left,x_proba,y_proba)
                    maximality_loop(x,y.right,x_proba,y_proba)
                    
                elif y.node.count() == 1 :
                    for i in range(0,nbData):
                        x_proba[i,0] *= x.node.proba[i][0]
                        x_proba[i,1] *= x.node.proba[i][1]
                    
                    maximality_loop(x.left,x.right,x_proba,x_proba.copy())
                    maximality_loop(x.left,y,x_proba,y_proba)
                    maximality_loop(x.right,y,x_proba,y_proba)
                    
                else :
                    raise Exception('Unexpected error')
                    
            else: #both children are not singletons
                for i in range(0,nbData):
                    x_proba[i,0] *= x.node.proba[i][0]
                    x_proba[i,1] *= x.node.proba[i][1] 
                    y_proba[i,0] *= y.node.proba[i][0]
                    y_proba[i,1] *= y.node.proba[i][1]
                      
                maximality_loop(x.left,x.right,x_proba,x_proba.copy())
                maximality_loop(y.left,y.right,y_proba,y_proba.copy())                    
                maximality_loop(x,y.left,x_proba,y_proba)
                maximality_loop(x,y.right,x_proba,y_proba)
                maximality_loop(x.left,y,x_proba,y_proba)
                maximality_loop(x.right,y,x_proba,y_proba)

                 
        maximality_loop(self.left,self.right,self.node.proba.copy(),self.node.proba.copy())
        return maximality_class
    
    
    def printTree(self, _p = 0):
        '''
        Method for printing a binary tree
        '''
        
        if _p == 0 :
            print self.node.label
            
        _p += 1
        
        if self.left is not None:
            print "    " * _p + str(self.left.node.label)
            self.left.printTree(_p) 
        
        if self.right is not None:
            print "    " * _p + str(self.right.node.label)
            self.right.printTree(_p) 
            
    def printProba(self, i = 0,_p = 0):
        '''
        Method for printing a binary tree
        '''
        
        if _p == 0 :
            print self.node.proba[i]
            
        _p += 1
        
        if self.left is not None:
            print "    " * _p + str(self.left.node.proba[i])
            self.left.printProba(i,_p) 
        
        if self.right is not None:
            print "    " * _p + str(self.right.node.proba[i])
            self.right.printProba(i,_p) 



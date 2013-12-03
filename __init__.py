from . import *
from classifip.dataset import arff


def count_labels(learndataset):
    """
    Computing the occurrences of each class values
    
    :return: dictionary of occurrences for class values
    :rtype: dict
    """
    occurrences = {}        
    for item in learndataset.data:
        class_value = item[-1]
        if occurrences.has_key(class_value): 
            occurrences[class_value] += 1
        else :
            occurrences[class_value] = 1
    return occurrences


def select_class_binary(learndataset, positive, negative):
    """return an ARFF object where only some classes are selected in order
    to form a dataset for a binary classification problem.
    
    :param select: the names of the classes to retain
    :type select: list
            
    :param positive: classes values to be considered as 'positive' in the binary
    classfication problem
    :type positive:list
    
    :param positive: classes values to be considered as 'positive' in the binary
    classfication problem
    :type positive:list
    
    :returns: the dictionary ('positive'/'negative') of selected data's 
    indices and a new ArffFile structure containing only selected classes
    :rtype: list of strings, :class:`~classifip.dataset.arff.ArffFile`
    
    .. warning::
    
        should not be used with files containing ranks
    """
    if 'class' not in learndataset.attribute_data.keys():
        raise NameError("Cannot find a class attribute.")
    if set(positive) - set(learndataset.attribute_data['class'])!=set([]):
        raise NameError("Specified 'positive' classes not a subset of existing ones!")
    
    if set(negative) - set(learndataset.attribute_data['class'])!=set([]):
        raise NameError("Specified 'negative' classes not a subset of existing ones!")
    
    selection=arff.ArffFile()
    #initiate the variable for the output data
    selected_data=[] 
    
    
    #construct list with sets of indices matching class names
    #assume the class is the last provided item
    #and form data corresponding to provided class
    for i,val in enumerate(learndataset.data) :

        if val[-1] in positive :
            buff = learndataset.data[i][:]
            buff[-1] = 'positive'
            selected_data.append(buff)

        elif val[-1] in negative:
            buff = learndataset.data[i][:]
            buff[-1] = 'negative'
            selected_data.append(buff)

    selection.attribute_data=learndataset.attribute_data.copy()
    selection.attribute_types=learndataset.attribute_types.copy()
    selection.attribute_data['class']=['positive','negative']
    selection.data = selected_data
    selection.relation=learndataset.relation
    selection.attributes=learndataset.attributes[:]
    selection.comment=learndataset.comment[:]

    return selection
''' Author: Thomas Waters '''

from task_1 import parseFile
import collections
import numpy as np
from math import log
from copy import copy

## - - - - Node Classes - - - - - - - -

class Node():
    '''     This is an abstract class
        The node classes contains all the information about a node
        in the decision tree. This includes parent/child nodes, split
        information, and terminal states.
    '''

    def __init__(self, parent=None):
        self.parent = parent

    def get_result(self, sample):
        ''' Abstract function
            Args:
                sample (1D numpy array): The data sample to be analized
            Returns:
                the predicted result of the sample based
                on the decission tree
            Raises:
                NotImplemenetedError
        '''
        raise NotImplementedError

class ContinuousNode(Node):
    ''' The continuous node is a node in the the tree that makes
        a decision on continuous or real data
        This results in having a binary split between two parts
        of the data resulting in two children nodes
    '''

    def __init__(self, attribute, split_val, parent=None):
        Node.__init__(self, parent)
        self.attribute = attribute
        self.split_val = split_val
        self.left_child = None
        self.right_child = None

    def get_result(self, sample):
        value = sample[self.attribute]
        if value < self.split_val:
            return self.left_child.get_result(sample)
        else:
            return self.right_child.get_result(sample)

class DiscreteNode(Node):
    ''' The discrete node is a node in the tree that makes a
        decision on discrete data
        This results in having a split in the data for each of the
        discrete elements in the particular data set
    '''

    def __init__(self, attribute, current_result, parent=None):
        Node.__init__(self, parent)
        self.current_result = current_result
        self.attribute = attribute
        self.children = {}

    def get_result(self, sample):
        value = sample[self.attribute]

        if value in self.children:
            return self.children[value].get_result(sample)
        else:
            return self.current_result

class LeafNode(Node):
    ''' The leaf node is a terminal node in the decision graph
        It holds its parent object and the resulting class from
        the decisions
    '''

    def __init__(self, result, parent):
        Node.__init__(self, parent)
        self.result = result

    def get_result(self, sample):
        return self.result

## - - - - Decision Tree Class - - - - -

class DecisionTree():
    ''' This is the object that holds the graph of the desiciion tree
        It is responsible for creating the tree from the initial training set
    '''

    def __init__(self):
        ## Terminal Conditions
        self.min_num_records = 20
        self.min_allowed_ig = 0.1        

    def build_tree(self, training_data, attribute_type):
        ''' the function then builds the decision tree by calling the
            self.create_node to build the decision tree on the data
            
            Args:
                training_data (nxm dimension numpy array): input training data where
                    n is the number of records in the data set
                    m is the number of parameter arguments
                attribute_type (list): a list of length m that contains
                    the number of unique elemets per argument
                    or -1 if the argument is continuous
        '''
        
        self.num_records = training_data.shape[0]
        self.num_attributes = training_data.shape[1]
        self.attribute_type = attribute_type

        ## Construct the decision tree
        searchable_attributes = [i for i in range(self.num_attributes - 1)]
        self.root = self.create_node(training_data, searchable_attributes, attribute_type)

    def predict(self, sample):
        ''' Args:
                sample (numpy array): data sample to be predicted using
                    the decision tree
            Returns:
                Takes a numpy array of 1xn where n is the number of attributes
                and returns the prediction for a given data sample
        '''
            
        return self.root.get_result(sample)

    def get_predictions(self, data):
        ''' Args:
                data(numpy array): the entire set of data to be predicted
            Returns:
                list: a set of data containing a prediction for each data sample
        '''
        predictions = []

        for trial in data:
            result = self.predict(trial)
            predictions.append(result)

        return predictions

    def create_node(self, training, searchable_attributes, attribute_type, parent=None):
        ''' This is the recursive function that takes care
            of creating the nodes of the tree
            It creates the nodes that has the best information gain
            and then recursively creates the children nodes until
            the terminal condition is reached
            
            Args:
                training (numpy array): The data set to use for predicting the current tnode
                searchable_attributes (list<int>): A set containing the attribute numbers that can be used
                    for the current node in the cree
                attribute_type (list<int>): A set containing the number of discrete values for each attribute
                    this number is -1 if the particular attribute is continious
                parent (Node): The parent node in the tree for where the current node is being created
                    
            Returns:
                Node: the node node in the tree that gives the most information gain
                    given the input training data
        '''

        current_num_records = training.shape[0]
        class_attributes = training[:, -1]
        current_result = collections.Counter(class_attributes).most_common()[0][0]

        ''' --- Test Terminal Conditions --- '''
        
        ## Test to see if all attributes are the same
        all_same = True
        for i in range(class_attributes.size - 1):
            if class_attributes[i] != class_attributes[i+1]:
                all_same = False
                break

        ## Or number of elements < threshold
        if all_same or current_num_records < self.min_num_records:
            return LeafNode(current_result, parent)


        ''' --- Calculate the entropy for each attribute '''

        y = training[:,-1:]

        entropy_dict = {}
        
        ## Used for the continuous values
        split_vals = {}
        
        for i in searchable_attributes:
            
            x = training[:, i:i+1]

            ## Currently is not counting the starting entropy
            ## Could add a terminal condition for the ammound of information gain
            ## starting_entropy = entropy(y)
            
            ## Is continuous
            if attribute_type[i] == -1:           
                cont_entropy, split  = split_real(x, y)
                entropy_dict[i] = cont_entropy
                split_vals[i] = split

            ## Is discrete
            else:
                entropy_dict[i] = conditional_entropy(x, y)

        best_attribute = min(entropy_dict, key=entropy_dict.get)
        new_searchable_attributes = copy(searchable_attributes)
        new_searchable_attributes.remove(best_attribute)

        ''' --- Test Terminal Condition --- '''

        ## Max information gain < threshold
        max_ig = entropy(y) - entropy_dict[best_attribute]
        if max_ig < self.min_allowed_ig:
            return LeafNode(current_result, parent)
        
        
        ''' --- Recursively Build the Decision Tree --- '''

        node = None
        
        ## Best node is continuous
        if attribute_type[best_attribute] == -1:
            split = split_vals[best_attribute]
            node = ContinuousNode(best_attribute, split, parent)

            ## Break up the data into two left and right branches
            lower_data = []
            greater_data = []
            
            for i in range(current_num_records):
                child_record = training[i]
                
                if training[i, best_attribute] < split:
                    lower_data.append(child_record)
                else:
                    greater_data.append(child_record)

            ## Convert lists to np arrays
            lower_data   = np.array(lower_data)
            greater_data = np.array(greater_data)

            ## Recurse down into the children nodes
            node.left_child  = self.create_node(lower_data,   new_searchable_attributes, attribute_type, node)
            node.right_child = self.create_node(greater_data, new_searchable_attributes, attribute_type, node)

        ## Best node is discrete    
        else:
            node = DiscreteNode(best_attribute, current_result, parent)

            ## Branch records are set up improperly

            ## Calculate each branch node
            branch_records = {}
            for i in range(current_num_records):
                attribute_title = training[i, best_attribute]
                child_record = training[i]
                
                if attribute_title not in branch_records:
                    branch_records[attribute_title] = []
                
                branch_records[attribute_title].append(child_record)

            ## Create each branch node
            for attribute_title, node_data in branch_records.iteritems():
                node_array = np.array(node_data)
                node.children[attribute_title] = self.create_node(node_array, new_searchable_attributes, attribute_type, node)

        return node

## - - - - Entropy Calculations - - - -
''' Code for calculating entropies is based off of
    the information from
    http://www.saedsayad.com/decision_tree.htm
'''

def split_real(x, y):
    ''' Args:
            x (nx1D numpy array): a nx1 dimensional numpy array
            y (nx1D numpy array): a nx1 dimensional numpy array
        Returns:
            the smallest entropy of y conditional to x,
            and the location ot which the best split occures
    '''

    data = np.hstack((x, y))
    sorted_data = np.sort(data, axis=0)

    num_records = len(sorted_data)

    split_entropies = {}

    for i in range(num_records - 1):
        
        p1 = sorted_data[i]
        p2 = sorted_data[i+1]
        x1 = p1[0]
        x2 = p2[0]
        y1 = p1[1]
        y2 = p2[1]

        ## If the classes are the same do not check the split point
        if (y2 == y1):
            continue
        
        ## Calculate the weighted entropy with the current split value
        split_point = x1 + (x2 - x1) / 2.0

        left_split = y[:i]
        right_split = y[i:]

        l_entropy = entropy(left_split)
        r_entropy = entropy(right_split)

        ## Weighted entropy is the average of the left and right entropies
        ## That are both scaled by their number of elements
        weighted_entropy = (i/float(num_records)) * l_entropy + \
                           ((num_records-i)/float(num_records)) * r_entropy

        split_entropies[split_point] = weighted_entropy
        
    ## Pick and return the best split value and return the split point
    best_split = min(split_entropies)
    
    return split_entropies[best_split], best_split
        
def entropy(y):
    ''' Args:
            y (nx1D numpy array): a nx1 dimensional numpy array
        Returns:
            the entropy of a the given array        
    '''

    num_records = y.size

    attributes_count = np.unique(y, return_counts=True)
    num_attribute_values = len(attributes_count[0])

    ## Sum up the total entropy
    total_entropy = 0
    for i in range(num_attribute_values):
        attribute = attributes_count[0][i]
        count = attributes_count[1][i]

        ## Entropy = -pi log2 pi
        probability = count / (num_records * 1.0)
        entropy = -probability * log(probability, 2)
        total_entropy += entropy

    return total_entropy


def conditional_entropy(x, y):
    ''' Args:
            x (nx1D numpy array): a nx1 dimensional numpy array
            y (nx1D numpy array): a nx1 dimensional numpy array
        Returns:
            the entropy of y conditional to x
    '''
    
    num_records = y.size

    ## attributes_count is of the form
    ## { 'a': {class1 : count, class2: count, total: count}, 'b': {...}, ...}
    attributes_count = {}

    ## Gather attribute counts
    for i in range(num_records):

        data_class = x[i][0]
        data_result = y[i][0]

        if data_class in attributes_count:
            if data_result in attributes_count[data_class]:
                attributes_count[data_class][data_result] += 1
            else:
                attributes_count[data_class][data_result] = 1
                
            attributes_count[data_class]['total'] += 1
            
        else:
            attributes_count[data_class] = {}
            attributes_count[data_class][data_result] = 1
            attributes_count[data_class]['total'] = 1

    ## Sum up the total entropy
    weighted_average = 0
    for data_class, count_dict in attributes_count.iteritems():
        
        total_entropy = 0
        num_elements = count_dict['total']
        
        for result_class, count in count_dict.iteritems():

            ## Entropy = -pi log2 pi
            probability = count / (num_elements * 1.0)
            entropy = -probability * log(probability, 2)
            total_entropy += entropy

        weighted_average += num_elements / float(num_records) * total_entropy

    assert weighted_average >= 0 and weighted_average <= 1

    return weighted_average

## - - - - Classification Predctions - - - -

def prediction_accuracy(data, predictions):
    ''' Args:
            data (2D numpy array): the tested data set
            predicitons (list): the class prediction for the given data set
        Returns:
            the number of misclassified elements and the percent
            accuracy of the classification
    '''
    
    num_correct = 0

    for i in range(len(predictions)):
        if data[i][-1] == predictions[i]:
            num_correct += 1
            
    accuracy = num_correct / float(len(data)) * 100.0
    misclassified = len(data) - num_correct
    return misclassified, accuracy

## - - - - Functional Testing - - - - -

if __name__ == "__main__":

    data, attribute_type, train_data, test_data = parseFile('records.txt')

    decision_tree = DecisionTree()
    decision_tree.build_tree(train_data, attribute_type)

    train_predictions = decision_tree.get_predictions(train_data)
    test_predictions  = decision_tree.get_predictions(test_data)

    train_misclassified, train_accuracy = prediction_accuracy(train_data, train_predictions)
    test_misclassified,  test_accuracy  = prediction_accuracy(test_data, test_predictions)

    print "Decision Tree built"
    print
    print "Train Data"
    print "Misclassified:",train_misclassified, "out of", len(train_data)
    print "Accuracy:", train_accuracy
    print
    print "Test Data"
    print "Misclassified:", test_misclassified, "out of", len(test_data)
    print "Accuracy:", test_accuracy

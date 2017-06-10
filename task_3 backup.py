from task_1 import parseFile
import numpy as np
from math import log
from copy import copy
from scipy import stats

## This program requires the scipy module for calculating
## mode statistics to determine the leaf resultant values
## by calling scipy's stats.mode function

## - - - - Node Classes - - - - - - - -
''' The node classes contains all the information about a node
    in the decision tree. This includes parent/child nodes, split
    information, and terminal states. '''

class ContiniousNode():
    ''' The continious node is a node in the the tree that makes
        a decision on continious or real data
        This results in having a binary split between two parts
        of the data resulting in two children nodes
    '''

    def __init__(self, attribute, split_val, parent=None):
        self.parent = parent
        self.attribute = attribute
        self.split_val = split_val
        self.left_child = None
        self.right_child = None

    def get_result(self, sample):
        value = sample[self.attribute]
        if value < self.split_value:
            return self.left_child.get_result(sample)
        else:
            return self.right_child.get_result(sample)

class DiscreteNode():
    ''' The discrete node is a node in the tree that makes a
        decision on discrete data
        This results in having a split in the data for each of the
        discrete elements in the particular data set
    '''

    def __init__(self, attribute, current_result, parent=None):
        self.parent = parent
        self.current_result = current_result
        self.attribute = attribute
        self.children = {}

    def get_result(self, sample):
        value = sample[self.attribute]

        if value in self.children:
            return self.children[value].get_result(sample)
        else:
            return self.current_result

class LeafNode():
    ''' The leaf node is a terminal node in the decision graph
        It holds its parent object and the resulting class from
        the decisions
    '''

    def __init__(self, result, parent):
        self.parent = parent
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
        ''' args:
                training_data - a nxm dimension numpy array
                    n is the number of records in the data set
                    m is the number of parameter arguments
                attribute_type - a list of length m that contains
                    the number of unique elemets per argument
                    or -1 if the argument is continious
            
            the function then builds the decision tree by calling the
            self.create_node to build the decision tree on the data
        '''
        
        self.num_records = training_data.shape[0]
        self.num_attributes = training_data.shape[1]
        self.attribute_type = attribute_type

        ## Construct the decision tree
        searchable_attributes = [i for i in range(self.num_attributes - 1)]
        self.root = self.create_node(training_data, searchable_attributes, attribute_type)

    def predict(self, sample):
        ''' Takes a numpy array of 1xn where n is the number of attributes
            and returns the prediction for a given data sample
        '''
            
        return self.root.get_result(sample)

    def get_predictions(self, data):
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
        '''

        current_num_records = training.shape[0]
        class_attributes = training[:, -1]
        current_result = stats.mode(class_attributes)[0][0]        

        ## --- Test Terminal Conditions ---
        
        ## Test to see if all attributes are the same
        all_same = True
        for i in range(class_attributes.size - 1):
            if class_attributes[i] != class_attributes[i+1]:
                all_same = False
                break

        ## Or number of elements < threshold
        if all_same or current_num_records < self.min_num_records:
            return LeafNode(current_result, parent)

        ## --- End Terminal Conditions Test ---


        ## --- Calculate the entropy for each attribute

        y = training[:,-1:]

        entropy_dict = {}
        
        ## Used for the continious values
        split_vals = {}
        
        for i in searchable_attributes:
            
            x = training[:, i:i+1]

            ## Currently is not counting the starting entropy
            ## Could add a terminal condition for the ammound of information gain
            ## starting_entropy = entropy(y)
            
            ## Is continious
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

        ## --- Test Terminal Condition ---

        ## Max information gain < threshold
        max_ig = entropy(y) - entropy_dict[best_attribute]
        if max_ig < self.min_allowed_ig:
            return LeafNode(current_result, parent)

        ## --- End Terminal Condition Test ---
        
        ## This is where the tree is built recursively

        node = None
        
        ## Best node is continious or discrete
        if attribute_type[best_attribute] == -1:
            split = split_vals[best_attribute]
            node = ContiniousNode(split, best_attribute, parent)

            ## Break up the data into two branches
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
            node.left_node  = self.create_node(lower_data,   new_searchable_attributes, attribute_type, node)
            node.right_node = self.create_node(greater_data, new_searchable_attributes, attribute_type, node)
            
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

## - - - -

def split_real(x, y):

    data = np.hstack((x, y))
    sorted_data = np.sort(data, axis=0)

    num_records = len(sorted_data)

    split_entropies = {}

    for i in range(num_records - 2):
        
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

        weighted_entropy = (i/float(num_records)) * l_entropy + ((num_records-i)/float(num_records)) * r_entropy

        split_entropies[split_point] = weighted_entropy
        
    ## Pick and return the best split value and return the split point
    best_split = min(split_entropies)
    
    return split_entropies[best_split], best_split
        
def entropy(y):
    ''' arg, y - a nx1 dimensional numpy array
        returns the entropy of a the given array        
    '''

    num_records = y.size

    attributes_count = {}

    ## Gather attribute counts
    for data_point in y:

        val = data_point[0]

        if val in attributes_count:
            attributes_count[val] += 1
        else:
            attributes_count[val] = 1

    ## Sum up the total entropy
    total_entropy = 0
    for attribute, count in attributes_count.iteritems():

        ## Entropy = -pi log2 pi
        probability = count / (num_records * 1.0)
        entropy = -probability * log(probability, 2)
        total_entropy += entropy

    return total_entropy


def conditional_entropy(x, y):
    ''' args:
            x - a nx1 dimensional numpy array
            y - a nx1 dimensional numpy array
        returns the entropy of y conditional to x
    '''
    
    num_records = y.size
    
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
    ''' args:
            data - the tested data set
            predicitons - the class prediction for the given data set
        returns:
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

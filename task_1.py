''' Author: Thomas Waters '''

import numpy as np

# - - - - - Utility Functions - - - - -

def isNumber(s):
    ''' Args:
            s(string): the input string
        Returns:
            bool: true if string represents a number and can
            be converted into a float and false otherwise
    '''
    try:
        float(s)
        return True
    except ValueError:
        return False

# - - - - - - - - - - - - - - - - - - -
def parseFile(file_name):
    '''
        Args:
            file_name (string): the name of the data file to be parsed

        Returns:
            (numpy array, list<int>, numpy array, numpy array)
        
            numpy array    :  the converted data in a numpy array
            list<int>      :  A set containing the number of discrete values
                for each attribute in the dataset
                this number is -1 if the particular attribute is continious
            numpy array    : the subset of the converted used for training data
            numpy array    : the subset of the converted used for test data

    '''
    f = open('records.txt')

    num_attributes_table = 16   # as defined by the problem

    ## Used for creating the attribute_type array
    attributes_table = [{} for i in range(num_attributes_table)]
    

    list_records = []

    ## Read file to a 2D list
    for line in f:

        line_attributes_table = line.strip().split(',')

        ## Iterate over the line_attributes_table
        for i in range(num_attributes_table):

            if line_attributes_table[i] != '?':

                isNum = isNumber(line_attributes_table[i])

                ## Element is continious
                if isNum:
                    num_val = float(line_attributes_table[i])
                ## Element is discrete
                else:
                    ## If attribute has not been seen yet
                    if line_attributes_table[i] not in attributes_table[i]:
                        ## Set the numerical value to the number of values for
                        ## that attribute so far
                        attributes_table[i][line_attributes_table[i]] = len(attributes_table[i])

                    num_val = attributes_table[i][line_attributes_table[i]]
                
                line_attributes_table[i] = num_val

        list_records.append(line_attributes_table)

    ## Get an array of the length of the attributes_table
    num_records = len(list_records)
    attribute_type = np.zeros( (num_attributes_table, 1) )
    for i in range(num_attributes_table):
        if len(attributes_table[i]) == 0:
            attribute_type[i] = -1
        else:
            attribute_type[i] = len(attributes_table[i])

    ## Determine the average and
    ## Fill in the missing (?) data
    for i in range(num_attributes_table):
        data_col = [row[i] for row in list_records]
        
        ## Determine the average
        average = None

        ## Value is nominal
        if attribute_type[i] == -1:
            count = 0
            for val in data_col:
                if val != '?':
                    count += val
                    
            average = count / num_records

        ## Value is not nominal
        else:
            count = {}
            for val in data_col:
                if val != '?':
                    if val in count:
                        count[val] += 1
                    else:
                        count[val] = 1
                        
            average = max(count)

        ## Fill in all the missing (?) data
        for n in range(num_records):
            if data_col[n] == '?':
                list_records[n][i] = average

    ## Convert 2D list to a numpy array and shuffle
    data = np.array(list_records)
    np.random.shuffle(data)
    
    ## Split the array into 2 sets 80% training, 20% testing
    split_ratio = 0.8
    
    train_data, test_data = split_data(data, split_ratio)

    return data, attribute_type, train_data, test_data

def split_data(data, split_ratio):
    ''' Splits the input data set into two windows according to
        the split ratio
        
        Args:
            data (2D numpy array): input data set
            split_ratio(int): the ratio of left to right subsets
        Returns:
            the left hand window of the dataset
                according to the split ratio,
            the right hand window of the dataset
                according to the split ratio,
    '''

    num_records = data.shape[0]
    len_train_data = int(num_records * split_ratio)

    left_split = data[:len_train_data]
    right_split = data[len_train_data:]

    return left_split, right_split

    

## - - - - Functional Testing - - - - -

if __name__ == "__main__":

    data, attribute_type, train_data, test_data = parseFile('records.txt')

    print "Data"
    print data
    print "Data Shape"
    print data.shape
    print "Train and Test Data Shape"
    print "Train : ", train_data.shape
    print "Test  : ", test_data.shape
    print "Attribute Types"
    print attribute_type
    

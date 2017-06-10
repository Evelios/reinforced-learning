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

def strToNum(s):
    '''
        Converts any string to an integer representation

        If the string represents a number, then the string
        returns the float data of that string

        Otherwise it converts the string into an integer value
        if the string is a '-', the result is 0
        if the string is a '+', the result is 1
        otherwise the result is the sum of all the ascii values of
        the characters in the string

        This method does allow for two strings to return the same value
        eg.// "bba" and "aab" would return the same value
        because the function does not take order into account
    '''
    try:
        return float(s)
    except:
        if s == '-':
            return 0
        elif s == '+':
            return 1
        total = 0
        for c in s:
            total += ord(c)
        return total

# - - - - - - - - - - - - - - - - - - -
def parseFile(file_name):
    '''
        Args:
            file_name (string): the name of the data file to be parsed

        Returns:
            the converted data in a numpy array,
            the attribute types of the data,
            the subset of the converted used for training data,
            the subset of the converted used for test data,

    '''
    f = open('records.txt')

    num_attributes = 16   # as defined by the problem

    ## Used for creating the attribute_type array
    attributes_count = [[] for i in range(num_attributes)]
    

    list_records = []

    ## Read file to a 2D list
    for line in f:

        line_attributes = line.strip().split(',')

        ## Iterate over the line_attributes
        for i in range(num_attributes):

            if line_attributes[i] != '?':

                num_val = strToNum(line_attributes[i])
                isNum = isNumber(line_attributes[i])
                line_attributes[i] = num_val
                
                
                ## Convert all nominal values to integers
                if not isNum and \
                   num_val not in attributes_count[i]:
                    attributes_count[i].append(num_val)
                    
                    

        list_records.append(line_attributes)

    ## Get an array of the length of the attributes
    num_records = len(list_records)
    attribute_type = np.zeros( (num_attributes, 1) )
    for i in range(num_attributes):
        if len(attributes_count[i]) == 0:
            attribute_type[i] = -1
        else:
            attribute_type[i] = len(attributes_count[i])

    ## Determine the average and
    ## Fill in the missing (?) data
    for i in range(num_attributes):
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
    

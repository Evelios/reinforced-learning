''' Author: Thomas Waters '''

from task_1 import parseFile
import math

''' Some of the program code is based off of
    Jason Brownlee's code from the following link
    http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
'''


## - - - - Calculate Probability Statists - - -

def mean(numbers):
    ''' Args:
            numbers (list): input set of numbers
        Returns:
            the average or mean of the dataset
    '''
    return sum(numbers) / float(len(numbers))

def stdev(numbers):
    ''' Args:
            numbers (list): input set of numbers
        Returns:
            the standard deviation of the dataset
    '''
    avg = mean(numbers)
    variance = sum([pow(x-avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)

## - - - - Seperate By Class - - - - - -

def seperateByClass(dataset):
    ''' Args:
            dataset (numpy array): a 2D array containing the data
        Returns:
            This function seperates all the data according to the class
            that the data belongs to
            It returns a dictionary that has the length of number of result
            classes that the data has
    '''
    
    seperated = {}
    for row in dataset:
        dataClass = row[-1]
        if (dataClass not in seperated):
            seperated[dataClass] = []
        seperated[dataClass].append(row)
    return seperated

## - - - - Summarize Data Set - - - - -

def summarize(dataset, attribute_type):
    ''' Args:
            dataset (numpy array): the set of data to be summarized
            attribute_type (list<int>): A set containing the number of discrete values
                for each attribute in the dataset
                this number is -1 if the particular attribute is continious
        Returns:
            list: A summary for each attribute containing information used to calculate
            future inputs to the bayesian model
            Each attribute is stored as its corresponding position between the dataset and the list
            Attributes that are continuous and discrete are stored differently
            continuous: Store an tuple of (mean, stdev)
            Discrete: Store a dictionary of the probability of an attribute
                eg:// {4: 0.2, 7: 0.5, 2: 0.3}
    '''
    
    zipped_data = zip(*dataset)

    summaries = []
    for i in range(len(zipped_data) - 1):

        attr_data = zipped_data[i]
        
        ## Attribute is continuous
        if attribute_type[i] == -1:
            summaries.append( (mean(attr_data), stdev(attr_data)) )
            
        ## Attribute is discrete
        else:
            ## Count the number of attributes of each type
            probabilities = {}
            num_records = len(attr_data)
            
            for item in attr_data:
                if item not in probabilities:
                    probabilities[item] = 1
                else:
                    probabilities[item] += 1

            ## Normailize the probabilities
            for attribute in probabilities:
                probabilities[attribute] /= float(num_records)

            summaries.append(probabilities)

    return summaries

def summarizeByClass(train_data, attribute_type):
    ''' Args:
            train_data (numpy array): the training data set used to train
                the bayesian model
            attribute_type (list<int>): A set containing the number of discrete values for each attribute
                    this number is -1 if the particular attribute is continious

        Returns:
            dictinary(int, list): A summary of each possible class type besed on the parameters layed out in the
            summarize(...) function
    '''
    num_records = train_data.shape[0]
    num_attributes = train_data.shape[1]

    seperated = seperateByClass(data)

    summary = {}

    ## Generate the summaries of all of the classes
    for class_value, instances in seperated.iteritems():
        summary[class_value] = summarize(instances, attribute_type)

    return summary
            
## - - - - Classification Probabilities - - - -

def continuous_probability(sample, mean, stdev):
    ''' Args:
             sample (float): the tested data point
             mean (float): mean of the dataset being analised
             stdev (float): standard deviation of the dataset being analised
         Returns:
             int: the probability that sample belongs in the data set
    '''
    e = math.exp( -(math.pow(sample - mean, 2) / (2 * math.pow(stdev, 2)) ) )
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * e

def calculate_class_probabilities(summaries, trial, attribute_type):
    ''' Args:
            summaries (dictinary(int, list)): The summaries of the data from the
                summarizeByClass(...) function which serves is the bayesian model of the data
            trial (numpy array): a single data sample to be tested for class type
            attribute_type (list<int>): A set containing the number of discrete values for each attribute
                    this number is -1 if the particular attribute is continious
        Returns:
            dictionary (int, float): a dictionary containing the probabilities that given the
                trial data, that the trial data belongs to a particular class
    '''
    probabilities = {}
    for class_value, class_summaries in summaries.iteritems():
        probabilities[class_value] = 1

        for i in range(len(class_summaries)):
            sample = trial[i]
            
            ## continuous attribute
            if attribute_type[i] == -1:
                mean, stdev = class_summaries[i]
                probabilities[class_value] *= continuous_probability(sample, mean, stdev)
            ## Discrete attribute
            else:
                probability_dict = class_summaries[i]
                if sample not in probability_dict:
                    probabilities[class_value] *= 0
                else:
                    probabilities[class_value] *= probability_dict[sample]
    
    return probabilities
    

## - - - - Classification Predctions - - - -

def predict(summaries, trial, attribute_type):
    ''' Args:
             summaries (dictinary(int, list)): The summaries of the data from the
                summarizeByClass(...) function which serves is the bayesian model of the data
             trial (numpy array): a single data sample to be tested for class type
             attribute_type (list): a list containing the number of discrete
                 elements for a particular attribut, or -1 for continuous data

         Returns:
             the best class guess for a the trial given the trained summaries data
    '''
    
    probabilities = calculate_class_probabilities(summaries, trial, attribute_type)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.iteritems():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

def get_predictions(summaries, data, attribute_type):
    ''' Args:
            summaries (dictinary(int, list)): The summaries of the data from the
                summarizeByClass(...) function which serves is the bayesian model of the data
            data (numpy array): All of the data samples to be tested
            attribute_type (list): a list containing the number of discrete
                 elements for a particular attribut, or -1 for continuous data
        Returns:
            list(int): The set of all the predictions for the given input data
    '''

    predictions = []
    
    for trial in data:
        result = predict(summaries, trial, attribute_type)
        predictions.append(result)

    return predictions

## - - - - Classification Predctions - - - -

def prediction_accuracy(data, predictions):
    ''' Args:
            data (2D numpy array): the tested data set
            predicitons (list): the class prediction for the given data set
        Returns:
            (int, float): the number of misclassified elements and the percent
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

    summaries = summarizeByClass(train_data, attribute_type)

    train_predictions = get_predictions(summaries, train_data, attribute_type)
    test_predictions = get_predictions(summaries, test_data, attribute_type)

    train_misclassified, train_accuracy = prediction_accuracy(train_data, train_predictions)
    test_misclassified, test_accuracy = prediction_accuracy(test_data, test_predictions)
    
    print "Bayes Classifier built"
    print
    print "Train Data"
    print "Misclassified:",train_misclassified, "out of", len(train_data)
    print "Accuracy:", train_accuracy
    print
    print "Test Data"
    print "Misclassified:", test_misclassified, "out of", len(test_data)
    print "Accuracy:", test_accuracy

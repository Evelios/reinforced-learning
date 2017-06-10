
import collections, numpy

X = numpy.random.randint(4, size=(16,5))

print 'Dataset'
print X

print 'Column indexed 2 histogram'
print collections.Counter(X[:,0]).most_common()

print 'Most frequent values per column'
MF = [collections.Counter(X[:,i]).most_common()[0][0] for i in range(X.shape[1])]
print MF

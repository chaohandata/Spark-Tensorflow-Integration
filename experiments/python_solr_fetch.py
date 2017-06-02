# sample program to load data from solr and conver to numpy array
# Sanket Shahane

import pysolr
import numpy
import pandas as pd

def import_data_from_solr(url='http://localhost:8983/solr/mnist/'):
	solr = pysolr.Solr(url, timeout=10)
	results = solr.search('*:*', **{"fl": "field*"})
	df = pd.DataFrame(list(results))
	train=df.sample(frac=0.8,random_state=200)
	test=df.drop(train.index)
	X_train = train.drop(["field_1"],axis=1)	# fields_1 : class label
	y_train = train["field_1"]
	X_test = test.drop(["field_1"],axis=1)
	y_test = test["field_1"]
	return np.array(X_train),np.array(y_train),np.array(X_test),np.array(y_test)

X_train,y_train,X_test,y_test = import_data_from_solr()
print 'X_train:',X_train.shape,type(X_train)
print 'y_train:'y_train.shape
print 'X_test:',X_test.shape
print 'y_test',y_test.shape
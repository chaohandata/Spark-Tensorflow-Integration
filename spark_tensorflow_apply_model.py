import pyspark
import pysolr
import json
import h5py
import sys
import numpy as np
import pandas as pd
import itertools
import keras
from keras import optimizers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

sc = pyspark.SparkContext()
config = None
solr = None
load_url = ''
save_url = ''

def import_test_data_from_solr(url='http://localhost:8983/solr/mnist/'):
	global solr
	solr = pysolr.Solr(url, timeout=10)
	results = solr.search('*:*', **{"fl": "field*", "rows": "100"}) # check the query
	df = pd.DataFrame(list(results))
	df.drop(["field_1"],axis=1,inplace=True)
	return np.array(df).astype('float32')

def reconstruct_model_and_predict(data,id):
	# reconstruc the model from the broadcasted configuration
	model_reconstruction = Sequential.from_config(config.value['configuration'])
	model_reconstruction.set_weights(config.value['weights'])
	# Following step does not train the model again. Just compiles so it can predict in future
	model_reconstruction.compile(optimizer=config.value['optimizer'], loss=config.value['loss'])
	# make predictions
	p = model_reconstruction.predict(data.reshape(-1,data.shape[0]))
	# convert one hot encoding to numbers
	pp = p.argmax()
	row = '{"id":'+str(id)+',"prediction:":'+str(pp)+'}'
	# create a json object and write to solr immediately to avoid serial bottleneck later on
	solr = pysolr.Solr(save_url, timeout=10)
	solr.add([json.loads(row)])
	return p

def main(config_path,url = 'http://localhost:8983/solr/mnist/'):
	# load model configuration.
	config = np.load(config_path).item()
	# fetch data
	X_test = import_test_data_from_solr()
	X_test_rdd = sc.parallelize(X_test)
	# broadcast the model configuration
	global config
	config = sc.broadcast(config)
	# make the predictions and store in solr with index. That's why zipWithIndex is used
	predictions = X_test_rdd.zipWithIndex().map(lambda (x,i): reconstruct_model_and_predict(x,i))
	p = predictions.map(lambda x: x.argmax())
	return p

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print "Pass a json config file"
		
	else:
		json_object = json.loads(sys.argv[1])
		config_path = json_object["model_config_path"]
		global load_url
		load_url = json_object["load_url"]
		global save_url
		save_url = json_object["save_url"]
		print config_path,"\n",load_url,"\n",save_url
		preds = main(config_path,load_url)
		print preds.collect()

#global config_path
#config_path = '/Users/sanket/Desktop/dlhackweek/trained_model_config.npy'
#preds = main(model_path='/Users/sanket/Desktop/dlhackweek/trained_model/py',url='http://localhost:8983/solr/mnist/')
#global save_url
#save_url = ''
#preds.collect()

# Notes:
# python apply_model.py '/Users/sanket/Desktop/dlhackweek/trained_model_config.npy' 'http://localhost:8983/solr/mnist/' 'http://localhost:8983/solr/mnist_prediction/'
# export PYTHONPATH="$PYTHONPATH:/Users/sanket/home/fusion3.1/3.1.0-beta1/apps/spark-dist/python/lib"
# export SPARK_HOME="/Users/sanket/home/fusion3.1/3.1.0-beta1/apps/spark-dist/"
# Spark-Tensorflow-Integration
This project is under the Deep Learning Hack week. We have chosen to integrate the tensorflow with Lucidworks fusion.

We aim to leverage spark's parallel processing abilities and tensorflow's deep learning capabilites to choose the best set of hyperparameters.

1. Tensorflow when run on a single node uses all the available resources to train a model. However, often is the case that we need to choose a set of parameters like the number of neurons, learning rate etc. This is where spark plays its role. We can train multiple models simultaneously on multiple nodes.
2. Since, Spark supports python via the pyspark and py4j libraries it is pretty straight forward to deploy a script to train multiple models in a spark cluster.
3. Here we have shown a sample script which loads the famous MNIST digit recognition dataset indexed in Solr and a feed forward neural network architecture. We train multiple models on the spark cluster nodes with different set of parameters and save the best model
4. Finally in a separate job we use this saved model to apply on test dataset parallely using spark's parallel execution capabilites. The output is saved back to a Solr collection specified the json configuration file.

# Execution Instructions:
1. python spark_tensorflow_train_model.py '{"json":"configuration"}' Pass the json configuration file to the script. Sample file is given.
2. ./bin/spark-submit /path/to/spark_tensorflow_train_model.py '{"json":"configuration"}' to submit the file to the spark driver as a job.
3. This will train multiple models based on the set of hyperparameters choosen and will save the best model configuration to a file at location specified in the json configuration.

4. python spark_tensorflow_apply_model.py '{"json":"configuration"}' Pass the json configuration file to the script. Sample file is given. You need to pass the configuration file, load_url of a solr collection, save_url to save results to a solr collection.
5. ./bin/spark-submit /path/to/spark_tensorflow_apply_model.py '{"json":"configuration"}' to submit the file to the spark driver as a job.

# Issues you might face when submitting the script as a job:
It is common to see errors like no module pyspark, or SPARK_HOME not found.
To fix these issues on a mac/linux:
1. Go to $FUSION_HOME/apps/spark-dist/python/lib directory and extract the pyspark.zip and py4j.zip (names may vary slightly based on the version of fusion)
2. Open the terminal and issue the following commands"

export PYTHONPATH="$PYTHONPATH:/$FUSION_HOME/apps/spark-dist/python/lib"

export SPARK_HOME="$FUSION_HOME/apps/spark-dist/"
 
 E.g. paths: 
  
 export PYTHONPATH="$PYTHONPATH:/$FUSION_HOME/apps/spark-dist/python/lib"
 
 export SPARK_HOME="/Users/sanket/home/fusion3.1/3.1.0-beta1/apps/spark-dist/"
 
# Pending work and currently unsolved issues:
 Create an endpoint and submit a job via curl
 
 read/write data from solr as spark rdd. Lucidworks api support unknown.
 

# References:
http://machinelearningmastery.com/  
https://databricks.com  
https://github.com/maxpumperla/elephas

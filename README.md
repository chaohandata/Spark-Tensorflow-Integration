# Spark-Tensorflow-Integration
This project is under the Deep Learning Hack week. We have chosen to integrate the tensorflow with Lucidworks fusion.

We aim to leverage spark's parallel processing abilities and tensorflow's deep learning capabilites to choose the best set of hyperparameters.

1. Tensorflow when run on a single node uses all the available resources to train a model. However, often is the case that we need to choose a set of parameters like the number of neurons, learning rate etc. This is where spark plays its role. We can train multiple models simultaneously on multiple nodes.
2. Since, Spark supports python via the pyspark and py4j libraries it is pretty straight forward to deploy a script to train multiple models in a spark cluster.
3. Here we have shown a sample script which loads the famous MNIST digit recognition dataset and a feed forward neural network architecture. We train multiple models on the spark cluster nodes with different set of parameters.

# Execution Instructions:
1. The mnist_ffnn.py code can be copied and run the ./bin/pyspark console. It will finally give you the best trained model in a keras variable called bestModel. It can then be used to predict unseen records or save it to a file for later use.

2. The code can be also submitted as a job to spark driver. Use the command "./bin/spark-submit mnist_ffnn.py"

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
 Integrate the code with Fusion in such a way that a json job specification is submitted and it executes the job.
 
 When I navigate to the localhost:8767 I am not able to track the job! Is it the case that only scala jobs or jobs submitted via fusion are traked at the spark-master on local host.

# References:
http://machinelearningmastery.com/  
https://databricks.com  
https://github.com/maxpumperla/elephas

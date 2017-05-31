import tensorflow as tf
from keras import optimizers
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense

# prepare the data for parallel processing
data_rdd = spark.read.option("header",True).csv('/path/to/image1.csv')
train_rdd,test_rdd = data_rdd.randomSplit([0.9,0.1])

#test_rdd_bc = sc.broadcast(test_rdd)

train_rdd_x = train_rdd.select("0","1","2","3","4")
train_rdd_y = train_rdd.select("Class")

train_rdd_x_bc = sc.broadcast(train_rdd_x)
train_rdd_y_bc = sc.broadcast(train_rdd_y)

d = 5 # typically the shape of the input

def define_model(learning_rate, layer1_neurons, layer2_neurons): #(parameters)
	model = Sequential()
	model.add(Dense(layer1_neurons,input_dim = d, kernel_initializer='uniform', activation='sigmoid'))
	model.add(Dense(layer2_neurons, kernel_initializer='uniform', activation='sigmoid'))
	model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	adam = optimizers.Adam(lr = learning_rate, decay = 0.005)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(train_rdd_bc_x, train_rdd_bc_y, nb_epoch=10, batch_size=30, verbose=2)
	print 'training complete'
	return model

learning_rate = [0.01,0.05]
layer1_neurons = [10]
layer2_neurons = [4,6]
import itertools
all_experiments = list(itertools.product(learning_rate, layer1_neurons, layer2_neurons)) # all combinations of parameters
print(len(all_experiments))

num_nodes = 1
n = max(2, int(len(all_experiments) // num_nodes))
grouped_experiments = [all_experiments[i:i+n] for i in range(0, len(all_experiments), n)]
all_exps_rdd = sc.parallelize(grouped_experiments, numSlices=len(grouped_experiments))
results = all_exps_rdd.flatMap(lambda z: [define_model(*y) for y in z]).collect()














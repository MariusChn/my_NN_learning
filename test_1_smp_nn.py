#coding:GBK
import numpy as np
from neuralnetworklib import NeuralNetwork

#number of input, output, and hidden nodes
input_nodes=784
hidden_nodes=100
output_nodes=10     
#learning rate is 0.3
learning_rate=0.3
#create instance
n=NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
#load mnist training data
with open("mnist_train_100.csv")as training_data_file:
    training_data_list=training_data_file.readlines()
#train
#go through
for record in training_data_list:
    #split
    image_values=record.split(',')
    #scale and shift the inputs
    inputs=(np.asfarray(image_values[1:])/255*0.99)+0.01
    #create the target values
    targets=np.zeros(output_nodes)+0.01
    #set target value
    targets[int(image_values[0])]=0.99
    n.train(inputs,targets)

#query
with open("mnist_test_10.csv")as test_data_file:
    test_data_all=test_data_file.readlines()
test_data_0=test_data_all[0].split(',')
target_data=test_data_0[0]
print("my target data is:"+target_data)
input_query_data=np.asfarray(test_data_0[1:])/255*0.99+0.01
a=n.query(input_query_data)
print(a)

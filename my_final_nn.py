#coding:GBK
#neural network class defination
import numpy as np
import scipy.special

class NeuralNetwork:
    #initialise the neural network
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        #set number of nodes in each input, hidden layer
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        #learning rate
        self.lr=learningrate
        #weight metrix
        self.wih=np.random.normal(0.0,pow(self.hnodes,-0.5),\
        (self.hnodes,self.inodes))#初步的取-0.5到0.5之间的数，按照正态分布
        self.who=np.random.normal(0.0,pow(self.onodes,-0.5),\
        (self.onodes,self.hnodes))
        #activation function is the sigmoid function
        self.activation_function=lambda x:scipy.special.expit(x)
        pass
        
    #train the neural network
    def train(self,inputs_list,targets_list):
        #convert inputs to 2d array
        inputs=np.array(inputs_list,ndmin=2).T
        targets=np.array(targets_list,ndmin=2).T
        #calculate signals into hidden layer
        hidden_inputs=np.dot(self.wih,inputs)
        #calculate the signals emerging from hidden layer
        hidden_outputs=self.activation_function(hidden_inputs)
        #calculate signals into final output layer
        final_inputs=np.dot(self.who,hidden_outputs)
        #calculate the signals emerging from final output layer
        final_outputs=self.activation_function(final_inputs)
        
        #error is the (target-actual)
        output_errors=targets-final_outputs
        #hidden layer error is the output_errors, \
        #split by weights,recombined at hidden nodes
        hidden_errors=np.dot(self.who.T,output_errors)
        #这样，我们就拥有了我们所需要的一切
        
        #update the weights for the links between the hidden and output
        #layers
        self.who+=self.lr*np.dot((output_errors*final_outputs*\
        (1-final_outputs)),np.transpose(hidden_outputs))
        #update the weights for the links between the input and hidden
        #layers
        self.wih+=self.lr*np.dot((hidden_errors*hidden_outputs*\
        (1-hidden_outputs)),np.transpose(inputs))
        pass
        
    #query the neural network
    def query(self,inputs_list):
        #convert inputs list to 2d array
        inputs=np.array(inputs_list,ndmin=2).T
        #calculate signals into hidden layer
        hidden_inputs=np.dot(self.wih,inputs)
        #calculate the signals emerging from hidden layer
        hidden_outputs=self.activation_function(hidden_inputs)
        #calculate signals into final output layer
        final_inputs=np.dot(self.who,hidden_outputs)
        #calculate the signals emerging from final output layer
        final_outputs=self.activation_function(final_inputs)
        return final_outputs

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

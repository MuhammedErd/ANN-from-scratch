# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 21:52:21 2018

@author: muham
"""
#imorting libraries
import pandas as pd
import numpy as np
import math 


# Importing the dataset
dataset = pd.read_csv("input.csv", sep=';')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,64].values
bias = 1
noe = 500 # number of epoch 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#normalization
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)


## 
## ANN
##
#initialize the weights
w_n = 133  # number of weight
w = np.random.uniform(-0.7,0.7,[w_n])  # original weights 
uw = np.ndarray.copy(w)                # holds updated weights


c = 0 # counter for features and weights
i = 0 # example counter

LR = 0.6 #learning rate

trd = len(y_train)  # the size of the train data

###
###
### FORWARD PASS
###
###
while (noe > 0):  
    i = 0 # initializing for a new epoch
    c = 0 # initializing for a new epoch
    while (i< trd):
        
        outo1 = np.empty([trd]) #holding output values in matrix
        
        #first hidden neuron weight * inputs
        #the first 65 weights, including bias, goes towards first hidden neuron
        c = 0     #initializing for a new example
        neth1 = 0 # input value of first neuron
        while (c <64) :
                mul = w[c] * X_train[i,c]
                neth1 = neth1 + mul # adding every value that is weight * input
                c+=1
        mul = w[c] * bias #bias * weight
        neth1 = neth1 + mul 
        c+=1
        
        #end of the first hidden neuron
        
        #second hidden neuron weight * inputs
        #the weights between 65 and 130, including bias, goes towards second hidden neuron
        neth2 = 0 # input value of second neuron
        while (c <129) :
                mul = w[c] * X_train[i,(c-65)]
                neth2 = neth2 + mul # adding every value that is weight * input
                c+=1
        mul = w[c] * bias #bias * weight
        neth2 = neth2 + mul
        c+=1
        
        #end of the second hidden neuron
    
    
        # calculation of output values of hidden neurons
        outh1 = 1/(1 + (math.exp(-neth1)))
        outh2 = 1/(1 + (math.exp(-neth2)))
        
        #calculation of input value of output neuron
        neto1 = outh1 * w[c] + outh2 * w[c+1] + bias * w[c+2]
        
        #calculation of output value
        outo1[i] = 1/(1 + (math.exp(-neto1)))
        
        #calculation of error of output1
        eo1 = 0.5 * math.pow((y_train[i]-outo1[i]), 2)
        
        
        
        #end of the forward pass
        
        
        
        
        ###
        ###
        ###  BACKWARD PASS
        ###
        ###
        
        
        DOEO1 = outo1[i] - y_train[i] #derivative of Error wrt outo1
        DOO1N1 = outo1[i] * ( 1- outo1[i] ) #derivative of outo1 wrt neto1
        DOOH1NH1 = outh1 * (1 -outh1) #derivative of outh1 wrt neth1
        DOOH2NH2 = outh2 * (1 -outh2) #derivative of outh2 wrt neth2
        DONO1OH1 = w[130] #derivative of neto1 wrt outh1
        DONO1OH2 = w[131] #derivative of neto1 wrt outh2
        
        c = 0 # initializing  
        
        ##
        ## updating the weights connected to  hidden neuron 1
        ##      
        while (c <64) :
    
            DOEOH1 = DOEO1 * DOO1N1 * DONO1OH1 #derivative of Error  wrt outh1       
            DONH1W = X_train[i, c] #derivative of neth1 wrt weight
            DOEW = DOEOH1 * DOOH1NH1 * DONH1W #graident descent algo
            
            uw[c] = w[c] - LR * DOEW #holding updated weights in different array
            c+=1
        DONH1W = bias #derivative of neth1 wrt weight
        DOEW = DOEOH1 * DOOH1NH1 * DONH1W #graident descent algo
        uw[c] = w[c] - LR * DOEW #holding updated weights in different array
        c+=1
        
        
        ##
        ## updating the weights connected to hidden neuron 2
        ##              
        while (c <129) :
        
            DOEOH2 = DOEO1 * DOO1N1 * DONO1OH2 #derivative of Error  wrt outh2
            DONH2W = X_train[i, (c-65)] #derivative of neth2 wrt weight
            DOEW = DOEOH2 * DOOH2NH2 * DONH2W #graident descent algo    
            
            uw[c] = w[c] - LR * DOEW #holding updated weights in different array
            c+=1
        DONH2W = bias #derivative of neth1 wrt weight
        DOEW = DOEOH2 * DOOH2NH2 * DONH2W #graident descent algo
        uw[c] = w[c] - LR * DOEW #holding updated weights in different array
        c+=1
        
        
               
        ##
        ##  updating the weights connected to output neuron
        ##
        DON1W133 = bias #derivative of neto1 wrt weight133
        DOEW133 = DOEO1 * DOO1N1 * DON1W133 #graident descent algo
        uw[132] = w[132] - LR * DOEW133 #holding updated weights in different array
              
        DON1W132 = outh2 #derivative of neto1 wrt weight132
        DOEW132 = DOEO1 * DOO1N1 * DON1W132 #graident descent algo
        uw[131] = w[131] - LR * DOEW132 #holding updated weights in different array
                
        DON1W131 = outh1 #derivative of neto1 wrt weight131
        DOEW131 = DOEO1 * DOO1N1 * DON1W131 #graident descent algo
        uw[130] = w[130] - LR * DOEW131 #holding updated weights in different array
                
        i+=1
        w = np.ndarray.copy(uw) #at the end of the epoch, updated wieghts be writing on the old weights
                
 
    noe-=1
    print("remaining epoch: {0}" .format(noe))   
    
###
### Testing the ANN
###
    
ted = len(y_test)       # the size of the test data
outo1 = np.empty([ted]) #output values
y_pred = np.zeros(ted, dtype=int) # initialing an array for predicted values
     
i = 0 # initializing the counter for examples
while (i < ted):
        #first hidden neuron weight * inputs
        #the first 65 weight, including bias, goes towards first hidden neuron
        c = 0

        neth1 = 0 # input value of first neuron
    
        while (c <64) :
                mul = w[c] * X_test[i,c]
                neth1 = neth1 + mul # adding every value that is weight * input
                c+=1
        mul = w[c] * bias #bias multiplication
        neth1 = neth1 + mul
        c+=1
        
        
         #end of the first hidden neuron
        
        #second hidden neuron weight * inputs
        #the weights between 65 and 130, including bias, goes towards second hidden neuron
        neth2 = 0 # input value of second neuron
        while (c <129) :
                mul = w[c] * X_test[i,(c-65)]
                neth2 = neth2 + mul # adding every value that is weight * input
                c+=1
        mul = w[c] * bias #bias multiplication
        neth2 = neth2 + mul
        c+=1
        
        
         #end of the second hidden neuron
        
        
        # calculation of output values of hidden neurons
        outh1 = 1/(1 + (math.exp(-neth1)))
        outh2 = 1/(1 + (math.exp(-neth2)))
        
        #calculation of input value of output neuron
        neto1 = outh1 * w[c] + outh2 * w[c+1] + bias * w[c+2]
        
        #calculation of output
        outo1[i] = 1/(1 + (math.exp(-neto1)))
        
        
        #used 0.5 as threshold vulue
        if (outo1[i] > 0.5):
            y_pred[i] = 1
        else:
            y_pred[i] = 0
        
        i+=1


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()  

accuracy = (tp + tn)/ (tp + tn + fp + fn) # calculating accuracy
precision = tp / (tp + fp) # calculating precision
recall = tp / (tp + fn) # calculating recall
f1_score = 2 *((precision * recall) / (precision + recall)) # calculating f1-score
print("accuracy: {0}" .format(accuracy))

print("precision: {0}" .format(precision))

print("recall : {0}" .format(recall))

print("f1-score : {0}" .format(f1_score))


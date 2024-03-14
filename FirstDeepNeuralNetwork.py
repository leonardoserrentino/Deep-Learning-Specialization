import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases import *
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
from public_tests import *
%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
%load_ext autoreload
%autoreload 2
np.random.seed(1)

#Inizializza parametri di weights e bias generico
def initialize_parameters(n_x, n_h, n_y):
	np.random.seed(1)
    
    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
    
#Inizializza i parametri per tutti i layer
def initialize_parameters_deep(layer_dims):
	np.random.seed(3)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        parameters['W'+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
        parameters['b'+str(l)]=np.zeros((layer_dims[l],1))

        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    
    return parameters
    
#Forward Propagation lineare
def linear_forward(A, W, b):
    Z=np.dot(W,A)+b
    cache = (A, W, b)
    
    return Z, cache

#Forward propagation con differenziazione di tipo di funzione di attivazione
def linear_activation_forward(A_prev, W, b, activation):
	if activation == "sigmoid":
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=sigmoid(Z)
    
    elif activation == "relu":
        Z,linear_cache=linear_forward(A_prev,W,b)
        A,activation_cache=relu(Z)
    cache = (linear_cache, activation_cache)

    return A, cache
    
#Applicazione delle funzioni di forward per tutti i layer del modello
def L_model_forward(X, parameters):
	caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A 								    
        	A,cache=linear_activation_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],'relu')
        caches.append(cache)
      
	AL,cache=linear_activation_forward(A,parameters['W'+str(l+1)],parameters['b'+str(l+1)],'sigmoid')
    caches.append(cache)
          
    return AL, caches
    
#Calcola il costo del modello
def compute_cost(AL, Y):
    m = Y.shape[1]

    cost=-(np.dot(Y,np.log(AL).T)+np.dot((1-Y),np.log(1-AL).T))/m
	cost = np.squeeze(cost)
    
    return cost
    
#Generalizzazione della backward propagation
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW=np.dot(dZ,A_prev.T)/m
    db=np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev=np.dot(W.T,dZ)
    
    return dA_prev, dW, db
    
#Backward propagation con differenziazione della funzione di attivazione (derivata)
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ=relu_backward(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache)
        
    elif activation == "sigmoid":
        dZ=sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache)
    
    return dA_prev, dW, db
    
#Implementazione della backward propagation su tutto il modello
def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL=-(np.divide(Y,AL)-np.divide(1-Y,1-AL))

    current_cache=caches[L-1]
    dA_prev_temp,dW_temp,db_temp=linear_activation_backward(dAL,current_cache,'sigmoid')
    grads["dA"+str(L-1)]=dA_prev_temp
    grads["dW"+str(L)]=dW_temp
    grads["db"+str(L)]=db_temp
    
    for l in reversed(range(L-1)):
        current_cache=caches[l]
        dA_prev_temp,dW_temp,db_temp=linear_activation_backward(grads['dA'+str(l+1)],current_cache,'relu')
        grads["dA"+str(l)]=dA_prev_temp
        grads["dW"+str(l+1)]=dW_temp
        grads["db"+str(l+1)]=db_temp

    return grads
    
#Utilizzo ultimo della backward propagation per aggiornare i pesi del modello per essere più accurato    
def update_parameters(params, grads, learning_rate):
    parameters = params.copy()
    L = len(parameters) // 2

    for l in range(L):
        parameters['W'+str(l+1)]=parameters['W'+str(l+1)]-learning_rate*grads['dW'+str(l+1)]
        parameters['b'+str(l+1)]=parameters['b'+str(l+1)]-learning_rate*grads['db'+str(l+1)]
    
    return parameters


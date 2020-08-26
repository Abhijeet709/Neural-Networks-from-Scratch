import numpy
import pandas
class NeuralNetwork:
    def __init__(self):

    def relu(self,Z):
        A = np.maximum(0,Z)
        cache = Z
        return A,cache
    
    def sigmoid(self,Z):
        A = 1/(1+np.exp(-Z))
        cache = Z
        return A,cache

    def relu_backward(self,dA,cache):
        Z = cache
        dZ = np.array(dA,copy=True)
        dZ[Z<=0] = 0
        return dZ

    def sigmoid_backward(self,dA,cache):
        Z = cache
        s = 1/(1+np.exp(-Z))
        dZ = dA*s*(1-s)
        return dZ
        
    def getParameters(self,x,h,y):
        W1 = np.random.randn(h,x)
        b1 = np.zeros((h,1))
        W2 = np.rand.random.randn(y,h)
        b2 = np.zeros((y,1))
        para = {'W1':W1,'W2':W2,'b1':b1,'b2':b2}
        return para

    def getZ(self,A,W,b):
        Z = np.dot(A,W)+b
        cache = (A,W,b)
        return Z,cache


    def forwardLinearActivation(self,A_prev,W,b,activation):
        Z,linearCache = getZ(A_prev,W,b)
        if activation == 'sigmoid':
            A, activationCache = self.sigmoid(Z)
        else:
            A, activationCache = self.relu(Z)
        cache = (linearCache,activationCache)
        return A,cache

    def getCost(self,y,Y):
        m = Y.shape[1]
        cost = np.sum((-(Y*(np.log(y)))+((Y-1)*(np.log(y-1))))/m,axis=1,keepdims=True)
        cost = np.squeeze(cost)
        return cost

    def backward(self,dZ,cache):
        A_prev,W,b = cache
        m = A_prev.shape[1]
        dW = np.dot(dZ,np.transpose(A_prev))/m
        db = np.sum(b,axis=1,keepdims=True)/m
        dA_prev = np.dot(np.transpose(W),dZ)
        return dW,db,dA_prev

    def backwordLinearActivation(self,dA,cache,activation):
        linear_cache,activation_cache = cache
        if activation == 'relu':
            dZ = self.relu_backward(dA,activation_cache)
            dA_prev,dW,db = self.backward(dZ,linear_cache)
        else:
            dZ = self.sigmoid_backward(dA,activation_cache)
            da_prev,dW,db = self.backward(dA,linear_cache)
        return dA_prev,dW,db


    

    
    
    

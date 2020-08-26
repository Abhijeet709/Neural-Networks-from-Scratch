import numpy as np
class multiLayerNN:

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

    def initialiseParameter(self,layerDims):
        parameters = {}
        l = len(layerDims)
        for i in range(1,l):
            parameters['W'+str(i)] = np.random.randn(layerDims[i],layerDims[i+1])*0.01
            parameters['b'+str(i)] = np.zeros((layerDims[i],1))
            return parameters

    def forwardActivation(self,parameters,X):
        caches = []
        l = len(parameters)//2
        A=X
        for i in range(1,l):
            A_prev = A
            A,cache = self.relu(np.dot(parameters['W'+str(i)],A_prev)+ parameters['b'+str(i)])
            caches.append(cache)
        Al,cache = self.sigmoid(np.dot(parameters['W'+str(l)],A)+ parameters['b'+str(l)])
        caches.append(cache)
        return Al,caches

    def backwordLinearActivation(self,dA,cache,activation):
        linear_cache,activation_cache = cache
        if activation == 'relu':
            dZ = self.relu_backward(dA,activation_cache)
            dA_prev,dW,db = self.backward(dZ,linear_cache)
        else:
            dZ = self.sigmoid_backward(dA,activation_cache)
            da_prev,dW,db = self.backward(dA,linear_cache)
        return dA_prev,dW,db

    def backward(self,AL,Y,caches):
        grads = {}
        L = len(caches)
        m = AL.shape[1]
        dAL = -(np.divide(Y,AL)) - np.divide(1-Y,1-AL)
        current_cache = caches[-1]
        grads["dA"+str(L-1)],grads["dW"+str(L)],grads["db"+str(L)] = self.backwordLinearActivation(dAL,current_cache,activation="sigmoid")
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.backwordLinearActivation(grads["dA"+str(l+1)], current_cache,activation="relu")
            grads["dA"+str(l)] = dA_prev_temp
            grads["dW"+str(l+1)] = dW_temp
            grads["db"+str(l+1)] = db_temp
        return grads

    def updateParameters(self,parameters,grads,learningRate):
        L = len(parameters)//2
        for l in range(L):
            parameters["W"+str(l+1)] = parameters["W"+str(l+1)]-learningRate*grads["dW"+str(l+1)]
            parameters["b"+str(l+1)] = parameters["b"+str(l+1)]-learningRate*grads["db"+str(l+1)]
        return parameters
    

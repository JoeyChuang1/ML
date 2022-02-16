import pandas as pd
from pyparsing import Regex
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
# This file is for knn of dataset1
filename= '../data/newdata.csv'
data=pd.read_csv(filename)
data['U']=data['U'].replace(1,0)
data['U']=data['U'].replace(2,1)



np.random.seed(34245)

# Set features by typing column name:
x=np.array(data[['B','T']])

data=data.values
y=data[:,-1]
y=y.astype(np.int32)

(N,D), C = x.shape, np.max(y)+1
inds = np.random.permutation(N)
# Set training and testing data
x_train, y_train = x[inds[10:]], y[inds[10:]]
x_test, y_test = x[inds[:10]], y[inds[:10]]

euclidean = lambda x1, x2: np.sqrt(np.sum((x1 - x2)**2,axis = -1))
manhattan = lambda x1, x2: np.sum(np.abs(x1 - x2), axis=-1)

class KNN:

    def __init__(self, K=1, dist= euclidean):
        self.dist = dist                                                    
                                                                                 
        self.K = K
        return
    
    def fit(self, x, y):
        ''' Store the training data using this method as it is a lazy learner'''
        self.x = x
        self.y = y
        self.C = np.max(y)+1
        return self
    
    def predict(self, x_test):
        ''' Makes a prediction using the stored training data and the test data given as argument'''
        
        num_test = x_test.shape[0]
        
        
        # For each test data, find its distance to every train data. Then get an array of shape: [num_test, num_train]          
        distances = self.dist(self.x[None,:,:], x_test[:,None,:])
                                 
        

        
        knns = np.zeros((num_test, self.K), dtype=int)
        
        
        y_prob = np.zeros((num_test, self.C))
        
        for i in range(num_test):
            knns[i,:] = np.argsort(distances[i])[:self.K]  # Find the indexs of training data for the k smallest distance
            y_prob[i,:] = np.bincount(self.y[knns[i,:]], minlength=self.C) #Find the number of each label
        # Find the frequency of each label
        
        y_prob /= self.K                                                         
        return y_prob, knns
    
model = KNN(K=10,dist=manhattan)
y_prob, knns = model.fit(x_train, y_train).predict( x_test)
print('K =',model.K)
print('Number of test data:', x_train.shape[0])
print('Number of test data:', x_test.shape[0])
y_pred = np.argmax(y_prob,axis=-1)# For every test data, find the most frequent label
    
accuracy = np.sum(y_pred == y_test)/y_test.shape[0] 
print(f'accuracy is {accuracy*100:.1f}.')

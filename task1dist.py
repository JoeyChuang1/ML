import pandas as pd
from pyparsing import Regex
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

# Compare the distribution of features in different label for the 1st dataset.
# If for the 2nd dataset, then need to do a little change: Delete line 12,13, change filename variable, change 'U' into 'T' in line 15,16.
filename='newdata2.csv'
data=pd.read_csv(filename)
data['U']=data['U'].replace(1,0)
data['U']=data['U'].replace(2,1)

# lable0Data=data.loc[data['T']==0]
# lable1Data=data.loc[data['T']==1]



def plot_histogram(feature):
    x0=np.array(lable0Data[feature])
    x1=np.array(lable1Data[feature])
    
    plt.hist([x0,x1],label=['negative','positive'])
    
    plt.legend(loc='upper left')
    
    plt.title('Compare the distribution of feature '+feature+' in different label')
    plt.show()
    

    
plot_histogram('H')
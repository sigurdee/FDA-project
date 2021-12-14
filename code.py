#matplotlib inline
import math
import numpy as np # imports a fast numerical programming library
import scipy as sp # imports stats functions, amongst other things
import matplotlib as mpl # this actually imports matplotlib
import matplotlib.cm as cm # allows us easy access to colormaps
import matplotlib.pyplot as plt # sets up plotting under plt
import pandas as pd # lets us handle data as dataframes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import seaborn as sns

# sets up pandas table display
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)

import seaborn as sns # sets up styles and gives us more plotting options


# NAIVE BAYES METHOD

# First opening the csv file for the training data

load = pd.read_csv('FDA-project/TravelInsurancePrediction_edit.csv')

#print(train_data.shape) # Prints (1987 , 10)


train_data = load.to_numpy() #converts from panda dataframe tom numpy array

train_data = train_data[: , 1:] #Slices off the indecies

print(train_data.shape) #Prints (1987 , 9)

print (train_data[:5 , :]) #Prints the first 5 rows 


#Now I'm going to try making some histograms/graphs to display the info from the csv file

# 






# Using a built in Naive Bayes function

def GaussNB(load):

    train_x = load.drop(columns=['TravelInsurance'] , axis = 1)
    train_y = load['TravelInsurance']

    model = GaussianNB(var_smoothing=0.25)  # Testing here with different var_smoothings; around 0.25 gives highests acc

    # fit the model with the training data
    model.fit(train_x,train_y)

    # predict the target on the train dataset
    predict_train = model.predict(train_x)
    print('Target on train data',predict_train) 

    # Accuray Score on train dataset
    accuracy_train = accuracy_score(train_y,predict_train)
    print('accuracy_score on train dataset : ', accuracy_train)


#GaussNB(load)
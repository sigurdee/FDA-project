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

load = pd.read_csv('FDA-project/TravelInsurancePrediction.csv')

#print(train_data.shape) # Prints (1987 , 10)

'''

train_data = load.to_numpy() #converts from panda dataframe tom numpy array

train_data = train_data[: , 1:]

print(train_data.shape)

print (train_data[:5 , :])

'''

train_x = load.drop(columns=['TravelInsurance'] , axis = 1)
train_y = load['TravelInsurance']

model = GaussianNB()

load.head()
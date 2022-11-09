
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from numpy import arange
import matplotlib.pyplot as plt
import random

from pandas import read_csv
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV

import pandas_datareader as pdr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pandas.plotting import autocorrelation_plot
import seaborn as sns
import math
sns.set_style("whitegrid")

from sklearn.preprocessing import MinMaxScaler 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from matplotlib.widgets import Slider

from fast_ml.model_development import train_valid_test_split
from sklearn.metrics import mean_absolute_error as mae

# Data Prepping

## Reading into Dataframes

df = pd.read_csv("ETF Prices.csv")

## Backfilling/ Forwardfilling

df=df.fillna(method="ffill")
df=df.fillna(method="bfill")

## Remove irrelevant ETFs

df_ESG = pd.read_csv('ESGSCORES.csv', index_col=False,usecols=[0,2])
assetLabels_ESG = df_ESG['Symbol'].tolist()

df_assetLabels_ESG=pd.DataFrame(assetLabels_ESG,columns=["Symbol"])
df_assetLabels_ESG=df_assetLabels_ESG['Symbol']

data = df[df["fund_symbol"].isin(df_assetLabels_ESG)]


## Save df as csv for use by ML Algos

data.to_csv('data.csv',index=False)  


## Generate closing values and returns

df = pd.read_csv("data.csv")
print(len(df['fund_symbol'].unique()))
all_stocks = pd.DataFrame(columns=['price_date'])
symbols=df['fund_symbol'].unique()

for symbol in symbols:
    df1=df.loc[df['fund_symbol'] ==symbol]
    tmp_close = df1[["price_date","adj_close"]]
    tmp_close.set_index("price_date", inplace=True)
    tmp_close=tmp_close.rename(columns={"adj_close": symbol})
    all_stocks=all_stocks.merge(tmp_close, how='outer',on='price_date')
    
all_stocks=all_stocks.fillna(method="ffill")
all_stocks=all_stocks.fillna(method="bfill")

all_stocks.to_csv('all_stocks.csv',index=False)  

data=pd.read_csv("data.csv")
funds=data.fund_symbol.unique()
df_withreturns=pd.DataFrame()
for i in funds:
    df=data[data['fund_symbol']==i]
    df['Daily Return'] = df['adj_close'].pct_change()  
    df=df.dropna()
    df_withreturns=df_withreturns.append(df)
df_withreturns.head()

returns = pd.DataFrame(columns=['price_date'])
for symbol in funds:
    df1=df_withreturns.loc[df_withreturns['fund_symbol'] ==symbol]
    tmp_close = df1[["price_date","Daily Return"]]
    tmp_close.set_index("price_date", inplace=True)
    tmp_close=tmp_close.rename(columns={"Daily Return": symbol})
    returns=returns.merge(tmp_close, how='outer',on='price_date')
    
returns=returns.dropna()

returns.head()

returns.to_csv('returns.csv',index=False) 

df=pd.read_csv("all_stocks.csv")
data=pd.read_csv("data.csv")
funds=data.fund_symbol.unique()
print(data.shape)
df_withreturns=pd.DataFrame()
for i in funds:
    df=data[data['fund_symbol']==i]
    df['Daily Return'] = df['adj_close'].pct_change()  
    df=df.dropna()
    df_withreturns=df_withreturns.append(df)
df_withreturns.head()

df_withreturns.to_csv('df_withreturns.csv',index=False) 


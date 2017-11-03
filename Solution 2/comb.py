import numpy as np
import pandas as pd
import gc
import os
os.chdir('D:\\python\\zillow') # change directory
from datetime import datetime

model_1 = pd.read_csv('lgb-5-cat-20.csv')
model_2 = pd.read_csv('lgb-5-cat-22.csv')
model_3 = pd.read_csv('lgb-10-20.csv')
model_4 = pd.read_csv('lgb-10-22.csv')
base_model = pd.DataFrame(model_1)

arr1 = model_1.values[:,1:]
arr2 = model_2.values[:,1:]
arr3 = model_3.values[:,1:]
arr4 = model_4.values[:,1:]
w = 0.3
w2 = 0.3
newArr = (arr1*(1-w)+arr3*w)*w2+(arr2*(1-w)+arr4*w)*(1-w2)

y_pred=[]
for i,predict in enumerate(newArr[:,0]):
        y_pred.append(str(round(predict,4)))

for idx,c in enumerate(base_model.columns[base_model.columns != 'ParcelId']):
    base_model[c] = np.array(y_pred)

base_model.to_csv('lgb-comb.csv', index=False)

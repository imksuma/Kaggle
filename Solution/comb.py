import numpy as np
import pandas as pd
import gc
import os
os.chdir('D:\\python\\zillow') # change directory
from datetime import datetime

base_model = pd.read_csv('lgb-5-cat.csv')
comp_model = pd.read_csv('lgb-10.csv')

arr1 = base_model.values[:,1:]
arr2 = comp_model.values[:,1:]
w = 0.3
newArr = arr1*(1-w)+arr2*w

y_pred=[]
for i,predict in enumerate(newArr[:,0]):
        y_pred.append(str(round(predict,4)))

for idx,c in enumerate(base_model.columns[base_model.columns != 'ParcelId']):
    base_model[c] = np.array(y_pred)

base_model.to_csv('lgb-comb.csv', index=False)

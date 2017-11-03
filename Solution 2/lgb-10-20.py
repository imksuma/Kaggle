'''
   author : Ilham Kusuma
   email : ilham.suk@gmail.com
   
   base model
'''
import numpy as np
import pandas as pd
import gc
import os
os.chdir('D:\\python\\zillow') # change directory
import lightgbm as lgb
from datetime import datetime

import pickle
fname_properties = 'properties_2017.pickle' # currently the best

def load_pickle(fname):
    output = None
    with open(fname, 'rb') as input:
        output = pickle.load(input)
    return output

def group(df_, colg, colt):
    t_la = df_.groupby(colg).mean()[colt].reset_index()
    t_z = pd.DataFrame({colg:df_[colg].values})
    t_z = t_z.merge(t_la, how='left', on=colg)
    return t_z[colt].values

properties = load_pickle(fname_properties)
print('load train')
train = pd.read_csv('train_2016_v2.csv', parse_dates=["transactiondate"])
train2 = pd.read_csv('train_2017.csv', parse_dates=["transactiondate"])
frame = [train, train2]
train = pd.concat(frame)
del frame, train2; gc.collect()

train['month'] = train.transactiondate.dt.month + (train.transactiondate.dt.year - 2016)*12
train_df = train.merge(properties, how='left', on='parcelid')

del properties; gc.collect()

train_df = train_df[train_df.logerror>-0.4]
train_df = train_df[train_df.logerror<0.42]

x_train = train_df.drop(['parcelid', 'logerror','transactiondate'], axis=1)
col_lgb = x_train.columns.values    
y_train = train_df["logerror"].values.astype(np.float32)

d_train = lgb.Dataset(x_train, label=y_train)
params = {}
params['max_bin'] = 10
params['learning_rate'] = 0.01 # shrinkage_rate
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'l2_root'          
params['bagging_fraction'] = 0.5 # sub_row
params['bagging_freq'] = 5
params['num_leaves'] = 511       # num_leaf
params['max_depth'] = 10
params['sub_feature'] = 0.5
params['xgboost_dart_mode'] = True
params['min_data_in_leaf'] = 40
cv_result = lgb.cv(params, d_train, num_boost_round=70000,
                   early_stopping_rounds=50,
                   verbose_eval=10, 
                   show_stdv=False
                  )
cv_df = pd.DataFrame(cv_result)
    
num_boost_round_lgb = len(cv_df)
clf = lgb.train(params, d_train, num_boost_round=num_boost_round_lgb)

del x_train, y_train, d_train, train_df, train; gc.collect()

properties = load_pickle(fname_properties)
properties['month'] = [20]*len(properties)
parcelid = properties.parcelid
x_test = properties[col_lgb]

clf.reset_parameter({"num_threads":1})
strt = np.array(range(int(len(parcelid)/30000)))*30000
strt = np.append(strt, [len(parcelid)])
end = strt[1:]
strt = strt[:-1]
pred = np.ones(len(parcelid))
for s, e in zip(strt,end):
        print(s)
        pred[s:e] =  clf.predict(x_test[s:e])

del properties, x_test; gc.collect()
y_pred=[]
for i,predict in enumerate(pred):
        y_pred.append(str(round(predict,4)))

zeros = np.array([0]*len(pred), dtype=int)
output = pd.DataFrame({'ParcelId': parcelid,
        '201610': y_pred, '201611': y_pred, '201612': y_pred,
        '201710': y_pred, '201711': y_pred, '201712': y_pred})

cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
output.to_csv('lgb-10-20.csv', index=False)
	
'''
   author : Ilham Kusuma
   email : ilham.suk@gmail.com
   
   drop calculatedbathnbr, finishedsquarefeet12, regionidcounty, censustractandblock
'''
import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression as lr

os.chdir('D:\\python\\zillow') # change directory
from datetime import datetime
import pickle

def save_pickle(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(fname):
    output = None
    with open(fname, 'rb') as input:
        output = pickle.load(input)
    return output

print('reading properties')
properties = pd.read_csv('properties_2017.csv')
drop_list = ['censustractandblock']
properties.drop(drop_list, axis=1, inplace=True)

print('reduce some zeros from some continues feature')
properties['latitude'] = np.round(properties['latitude']/1000,0)
properties['longitude'] = np.round(properties['longitude']/1000,0)
properties['rawcensustractandblock'] = np.round(properties['rawcensustractandblock']/10,0)

print('simplify categories propertycountylandusecode')
pland = properties.propertycountylandusecode.value_counts()/len(properties)
pland = pland.index.values[pland<=1.051548e-02]
temp = properties['propertycountylandusecode'].values
for id in pland:
   temp[temp == id] = '070P'

properties['propertycountylandusecode'] = temp

print('simplify categories propertylandusetypeid')
pland = properties.propertylandusetypeid.value_counts()
pland = [id for id, val in zip(pland.index.values,pland.values) if val < 10000]
temp = properties['propertylandusetypeid'].values
for id in pland:
   temp[temp == id] = 31.0

properties['propertylandusetypeid'] = temp

print('add new diff feature')
properties['newtax'] = (properties.structuretaxvaluedollarcnt.values - properties.landtaxvaluedollarcnt)
#properties['newtax'] = np.log(properties['newtax']+np.abs(properties['newtax'].min()))
properties['newtax'] = properties['newtax'].astype(np.float32)
properties['newtax'].fillna(-2*properties['structuretaxvaluedollarcnt'].max(),inplace=True)

properties['newyear'] = 2017-properties.yearbuilt
properties['newyear'].fillna(-1,inplace=True)
properties['newyear'] = properties['newyear'].astype(np.int16)
properties['newyear'].fillna(-1,inplace=True)

properties['newarea'] = properties.calculatedfinishedsquarefeet - properties.lotsizesquarefeet
#properties['newarea'] = np.log(properties['newarea']+ np.abs(properties['newarea'].min()))
properties['newarea'] = properties['newarea'].astype(np.float32)
properties['newarea'].fillna(-2*properties['calculatedfinishedsquarefeet'].max(),inplace=True)

print('categorical')
categorical = ['airconditioningtypeid',
               'architecturalstyletypeid',
               'buildingclasstypeid',
               'buildingqualitytypeid',
               'fips',
               'heatingorsystemtypeid',
               'propertycountylandusecode',
               'propertylandusetypeid',
               'propertyzoningdesc',
               'rawcensustractandblock',
               'regionidcity',
               'regionidcounty',
               'regionidneighborhood',
               'regionidzip',
               'typeconstructiontypeid',
               ]
for c in categorical:
    properties[c].fillna(-1,inplace=True)
    lbl = LabelEncoder()
    lbl.fit(list(properties[c].values))
    properties[c] = lbl.transform(list(properties[c].values))
    properties[c] = properties[c].astype(np.int32)
'''
print('log')
log_list = ['finishedfloor1squarefeet',
            'calculatedfinishedsquarefeet',
            'finishedsquarefeet12',
            'rawcensustractandblock',
            'structuretaxvaluedollarcnt',
            'taxvaluedollarcnt',
            'landtaxvaluedollarcnt',
            'taxamount'
           ]
for c in log_list:
    arr = properties[c].values#[properties[c].values==0] = 1
    arr[arr==0] = 1
    properties[c] = np.log(arr)
'''
# fill nan and label encoder categoriacal feature
print('boolean feature')
one_val = ['decktypeid',
           'hashottuborspa',
           'poolcnt',
           'pooltypeid10',
           'pooltypeid2',
           'pooltypeid7',
           'storytypeid',
           'fireplaceflag',
           'taxdelinquencyflag'
           ]
for c in one_val:
    isnan = properties[c].isnull().values
    properties[c] = ~isnan

print('continues feature')
continues = ['basementsqft',
             'bathroomcnt',
             'bedroomcnt',
             'calculatedbathnbr',
             'finishedfloor1squarefeet',
             'calculatedfinishedsquarefeet',
             'finishedsquarefeet12',
             'finishedsquarefeet13',
             'finishedsquarefeet15',
             'finishedsquarefeet50',
             'finishedsquarefeet6',
             'fireplacecnt',
             'fullbathcnt',
             'garagecarcnt',
             'garagetotalsqft',
             'latitude',
             'longitude',
             'lotsizesquarefeet',
             'poolsizesum',
             'roomcnt',
             'threequarterbathnbr',
             'unitcnt',
             'yardbuildingsqft17',
             'yardbuildingsqft26',
             'yearbuilt',
             'numberofstories',
             'structuretaxvaluedollarcnt',
             'taxvaluedollarcnt',
             'assessmentyear',
             'landtaxvaluedollarcnt',
             'taxamount',
             'taxdelinquencyyear'			 
            ]
for c in continues:
    properties[c].fillna(-1,inplace=True)
    properties[c] = properties[c].astype(np.float32)

def group(df_, colg, colt):
    t_la = df_.groupby(colg).mean()[colt].reset_index()
    t_z = pd.DataFrame({colg:df_[colg].values})
    t_z = t_z.merge(t_la, how='left', on=colg)
    return t_z[colt].values

print('calculate distance from the center')
latz = group(properties, 'regionidzip', 'latitude')
lonz = group(properties, 'regionidzip', 'longitude')
latc = group(properties, 'regionidcity', 'latitude')
lonc = group(properties, 'regionidcity', 'longitude')
latf = group(properties, 'regionidneighborhood', 'latitude')
lonf = group(properties, 'regionidneighborhood', 'longitude')

properties['distz'] = np.sqrt((latz-properties.latitude.values)**2+(lonz-properties.longitude.values)**2)
properties['distc'] = np.sqrt((latc-properties.latitude.values)**2+(lonc-properties.longitude.values)**2)
properties['distf'] = np.sqrt((latf-properties.latitude.values)**2+(lonf-properties.longitude.values)**2)

print('mean tax each region (zip, city, neighborhood)')
properties['taxz'] = group(properties, 'regionidzip', 'taxvaluedollarcnt')
properties['taxc'] = group(properties, 'regionidcity', 'taxvaluedollarcnt')
properties['taxf'] = group(properties, 'regionidneighborhood', 'taxvaluedollarcnt')
'''
print('propotional feature')
properties['psqrtax'] = properties['calculatedfinishedsquarefeet'].values/properties['taxvaluedollarcnt'].values
properties['ptaxyear'] = properties['taxvaluedollarcnt'].values/properties['yearbuilt'].values
properties['psqryear'] = properties['calculatedfinishedsquarefeet'].values/properties['yearbuilt'].values
'''
print('mean sqft each region (zip, city, neighborhood)')
properties['sqrz'] = group(properties, 'regionidzip', 'calculatedfinishedsquarefeet')
properties['sqrc'] = group(properties, 'regionidcity', 'calculatedfinishedsquarefeet')
properties['sqrf'] = group(properties, 'regionidneighborhood', 'calculatedfinishedsquarefeet')

for c in properties.columns:
    print(c, properties[c].isnull().mean())
    arr = properties[c].values
    arr[arr==np.inf] = 0
    arr[arr==-np.inf] = 0

fname_properties = 'properties_v7_2017.pickle'

save_pickle(properties, fname_properties)
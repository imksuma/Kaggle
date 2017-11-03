'''
   author : Ilham Kusuma
   email : ilham.suk@gmail.com
   
   this script is used to do data cleaning and feature engineering.
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
'''
print('simplify categories airconditioningtypeid')
pland = properties.airconditioningtypeid.value_counts()
pland_i = pland.index.values[pland<=8000]
temp = properties['airconditioningtypeid'].values
for id in pland_i:
   temp[temp == id] = pland_i[0]

properties['airconditioningtypeid'] = temp

print('simplify categories architecturalstyletypeid')
pland = properties.architecturalstyletypeid.value_counts()
pland_i = pland.index.values[pland<=400]
temp = properties['architecturalstyletypeid'].values
for id in pland_i:
   temp[temp == id] = pland_i[0]

properties['architecturalstyletypeid'] = temp

print('simplify categories buildingclasstypeid')
pland = properties.buildingclasstypeid.value_counts()
pland_i = pland.index.values[pland<=4000]
temp = properties['buildingclasstypeid'].values
for id in pland_i:
   temp[temp == id] = pland_i[0]

properties['buildingclasstypeid'] = temp

print('simplify categories buildingqualitytypeid')
pland = properties.buildingqualitytypeid.value_counts()
pland_i = pland.index.values[pland<=4000]
temp = properties['buildingqualitytypeid'].values
for id in pland_i:
   temp[temp == id] = pland_i[0]

properties['buildingqualitytypeid'] = temp

print('simplify categories heatingorsystemtypeid')
pland = properties.heatingorsystemtypeid.value_counts()
pland_i = pland.index.values[pland<=4000]
temp = properties['heatingorsystemtypeid'].values
for id in pland_i:
   temp[temp == id] = pland_i[0]

properties['heatingorsystemtypeid'] = temp

print('simplify categories propertyzoningdesc')
zone_null = properties.propertyzoningdesc.isnull()
temp = properties['propertyzoningdesc'].values
temp[zone_null] = 'nan'
properties['propertyzoningdesc'] = temp
properties['propertyzoningdesc'] = [ii[0] for ii in properties['propertyzoningdesc'].values]
pland = properties.propertyzoningdesc.value_counts()
pland_i = pland.index.values[pland<=400]
temp = properties['propertyzoningdesc'].values
for id in pland_i:
   temp[temp == id] = 'I'

properties['propertyzoningdesc'] = temp

print('simplify categories rawcensustractandblock')
pland = properties.rawcensustractandblock.value_counts()
pland_i = pland.index.values[pland<=8000]
temp = properties['rawcensustractandblock'].values
for id in pland_i:
   temp[temp == id] = 603756

properties['rawcensustractandblock'] = temp
'''


###################################################################################
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

properties['newyear'] = 2017-properties.yearbuilt
properties['newyear'].fillna(-1,inplace=True)
properties['newyear'] = properties['newyear'].astype(np.int16)

properties['newarea'] = properties.calculatedfinishedsquarefeet - properties.lotsizesquarefeet
#properties['newarea'] = np.log(properties['newarea']+ np.abs(properties['newarea'].min()))
properties['newarea'] = properties['newarea'].astype(np.float32)

'''	
properties['diffbedbath'] = (properties.bedroomcnt - properties.bathroomcnt)
properties['diffbedbath'].fillna(-properties['diffbedbath'].max(),inplace=True)
properties['diffbedbath'] = properties['diffbedbath'].astype(np.int16)

properties['diffgarbed'] = (properties.bedroomcnt - properties.garagecarcnt)
properties['diffgarbed'].fillna(-properties.diffgarbed.max(),inplace=True)
properties['diffgarbed'] = properties['diffgarbed'].astype(np.int16)

properties['diffunitbath'] = properties.unitcnt - properties.bathroomcnt
properties['diffunitbath'].fillna(-properties['diffunitbath'].max(),inplace=True)
properties['diffunitbath'] = properties['diffunitbath'].astype(np.int16)
'''

#properties['yearbuilt'] = np.round(properties['yearbuilt'].values/10,0)
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
    number_of_values = np.unique(properties[c].values[~properties[c].isnull()])
    if len(number_of_values) <= 50:
       for n in number_of_values: properties[c+str(n)] = properties[c].values==n
    properties[c].fillna(-1,inplace=True)
    lbl = LabelEncoder()
    lbl.fit(list(properties[c].values))
    properties[c] = lbl.transform(list(properties[c].values))
    properties[c] = properties[c].astype(np.int32)

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

del lbl,arr,drop_list; gc.collect()

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

#properties['taxland'] = group(properties, 'propertycountylandusecode', 'landtaxvaluedollarcnt')
#properties['taxf'] = group(properties, 'propertycountylandusecode', 'structuretaxvaluedollarcnt')

print('propotional feature')
properties['psqrtax'] = properties['calculatedfinishedsquarefeet'].values/properties['taxvaluedollarcnt'].values
properties['ptaxyear'] = properties['taxvaluedollarcnt'].values/properties['yearbuilt'].values
properties['psqryear'] = properties['calculatedfinishedsquarefeet'].values/properties['yearbuilt'].values

print('mean sqft each region (zip, city, neighborhood)')
properties['sqrz'] = group(properties, 'regionidzip', 'calculatedfinishedsquarefeet')
properties['sqrc'] = group(properties, 'regionidcity', 'calculatedfinishedsquarefeet')
properties['sqrf'] = group(properties, 'regionidneighborhood', 'calculatedfinishedsquarefeet')

'''

t_la = properties.groupby(['propertylandusetypeid','regionidcounty']).mean()['landtaxvaluedollarcnt'].reset_index()
properties = properties.merge(t_la, how='left', on=['propertylandusetypeid','regionidcounty'])

t_la = properties.groupby(['propertylandusetypeid','regionidcounty']).mean()['taxamount'].reset_index()
properties = properties.merge(t_la, how='left', on=['propertylandusetypeid','regionidcounty'])
'''
'''
properties['ybz'] = group(properties, 'regionidzip', 'yearbuilt')
properties['ybc'] = group(properties, 'regionidcity', 'yearbuilt')
properties['ybf'] = group(properties, 'regionidneighborhood', 'yearbuilt')
'''
print('read train 2016 v2')
train = pd.read_csv('train_2016_v2.csv', parse_dates=["transactiondate"])
train['month'] = train.transactiondate.dt.month
#df_ts = pd.DataFrame({'month':train.month.values, 'logerror':train.logerror.values})

print('merge train and properties')
train_df = train.merge(properties, how='left', on='parcelid')

fname_train_df = 'train_df_v4.pickle'
fname_properties = 'properties_2017.pickle'

#save_pickle(train_df, fname_train_df)
save_pickle(properties, fname_properties)

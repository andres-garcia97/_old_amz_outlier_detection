### Test file
# Raw data: data_url = 'http://bit.ly/2cLzoxH'
# pd.read_csv("https://raw.githubusercontent.com/JackyP/testing/master/datasets/nycflights.csv", usecols=range(1,17))

import pandas as pd
from pandas.core.base import DataError

raw_data = {'asin': ['B01FV62O1U', 'B0178OV2DO', 'B084PNTB81'],
            'pkg_length': [5.0, 8.0, 3.0],
            'pkg_width': [8.0, 1.5, 2.3],
            'pkg_height': [3.2, 1.5, 3.0],
            'pkg_dimensional_uom': ['inches', 'cm', 'centimeters'],
            'pkg_weight': [3.2, 1.5, 3.0],
            'pkg_weight_uom': ['kg', 'pounds', 'pounds']
           }

X = pd.DataFrame(raw_data, columns= ['asin','pkg_length', 'pkg_width', 'pkg_height', 'pkg_dimensional_uom', 'pkg_weight', 'pkg_weight_uom'])

print("Antes de la transformacion: \n")
print(X)

# Dimensions
if bool({'cm', 'centimeter', 'centimeters'}.intersection(list(X.pkg_dimensional_uom.unique()))):
    X.loc[((X.pkg_dimensional_uom=='cm')|(X.pkg_dimensional_uom=='centimeter')|(X.pkg_dimensional_uom=='centimeters')), ['pkg_height', 'pkg_width', 'pkg_length']] *= 3
    X.loc[((X.pkg_dimensional_uom=='cm')|(X.pkg_dimensional_uom=='centimeter')|(X.pkg_dimensional_uom=='centimeters')), 'pkg_dimensional_uom'] = 'inches' 

# Weight
if bool({'kg', 'kilo', 'kilogram', 'kilograms'}.intersection(list(X.pkg_weight_uom.unique()))):
    X.loc[((X.pkg_weight_uom=='cm')|(X.pkg_weight_uom=='centimeter')|(X.pkg_weight_uom=='centimeters')), ['pkg_height', 'pkg_width', 'pkg_length']] *= 2
    X.loc[((X.pkg_weight_uom=='cm')|(X.pkg_weight_uom=='centimeter')|(X.pkg_weight_uom=='centimeters')), 'pkg_dimensional_uom'] = 'kilograms' 

X = X[['pkg_height', 'pkg_width', 'pkg_length']]

print("Despues de la transformacion: \n")
print(X)




# pattern = re.compile(r'^(B00){1}')

# print([1 if pattern.match(x) else 0 for x in raw_data['asin']])
# print(raw_data.loc[[1 if pattern.match(x) else 0 for x in raw_data['asin']]])
# print(len(raw_data))
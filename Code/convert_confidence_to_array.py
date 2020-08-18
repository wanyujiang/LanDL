# -*- coding: utf-8 -*-
"""
Created on Sat May 16 10:18:26 2020

@author: marti_cn
"""

import numpy as np
import rasterio
import os
import pandas as pd


confidence_path= os.path.abspath( os.path.join('..', 'Data', 'ELSUSv2', 'raw' ) )  
opath = 'D:/landslides/' 
confidence = rasterio.open(os.path.join(confidence_path,'confidence.tif') ).read(1).astype(int)

indexes= np.where(confidence==3)
# save to npz
# Format: rows (y), cols (x)
np.savez_compressed(opath+'confidence_idx.npz',  array1= np.array(indexes).T)

######################################

# Mirar archivos
#ipath= 'D:/landslides/old/'
#files= ['dem.csv.gz', 'land_cover.csv.gz', 'lithology_fraction_first_layer.csv.gz', 'target.csv.gz', 'elsus_climate.csv']

#for file in files:
#    df= pd.read_csv(ipath+file)
#    print(file)
#    print(df.shape)
#    print(df.columns)
#    for c in df.columns[2:]:
#        if (np.any(df[c])== 255):
#            print(c)
#            print(df[c].nunique())

    

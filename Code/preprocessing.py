"""
Script to convert input raster files of features into tables.

@author: marti_cn
"""


import numpy as np
import rasterio
import pandas as pd
import os
from matplotlib import pyplot as plt
import re

class FromImageToTable(object):
    def __init__(self, train = True):
        """
        Class that converts images into tables to be handled by Python.
        train= True if you want to preprocess the train data, otherwise set this to False.
        """
        self.train= train
        if (self.train):
          confidence_path= os.path.abspath( os.path.join('..', 'Data', 'ELSUSv2', 'raw' ) )
          #load indexes where confidence =3. File created with convert_confidence_to_array.py
          idll = np.load(os.path.join(confidence_path, 'confidence_idx.npz')) 
          idll = idll['array1'].T
          idll = (np.array(idll[0,:]), np.array(idll[1,:]) )
          self.cidx = idll
          
    @staticmethod
    def from_array_to_table(array, filter_nan= False):
        """
        Function that converts an array into a table
        filter_nan: True: filters np.nan values. False: Filters 255.
        Output: table with columns: y, x, value
        """
        h, w = array.shape[0], array.shape[1]
        table = pd.DataFrame(array, index= np.arange(h), columns= np.arange(w))
        table = table.reset_index()
        table.rename(columns={'index': 'y'}, inplace= True)
        table = pd.melt(table, id_vars= ['y'], value_vars= np.arange(w))
        table.columns= ['y', 'x', 'value']
        if filter_nan:
            table= table.dropna(axis=0) 
        else:
            table = table[table.value != 255]
        return table
    
    def confidence(self, array, dtype= np.int16):
        array1 = array.copy()
        empty_ds= 255*np.ones(array.shape, dtype= dtype) 
        # maintain only values of dataset where confidence =3, rest =255
        empty_ds[self.cidx]= array[self.cidx]
        array1= empty_ds
        return array1
        
    def get_climate(self, ipath, opath, ifilename= 'climate_phys_regions.tif', ofilename= "elsus_climate.csv.gz"):
        """
        Convert from climate raster to climate table. (the same as get_elsus_features)
        ipath: path of raster file
        opath: path where table (txt format) will be stored. 
        """
        print('Reading raster of climate...')
        dataset = rasterio.open( os.path.join(ipath, ifilename) ).read(1).astype(int) # to int because values are codes
        
        if (self.train):
            dataset=  self.confidence(dataset)
            print('Climate raster converted to numpy array.')
            
        # From numpy to table
        df = self.from_array_to_table(dataset)
        df.rename(columns={'value': 'climate'}, inplace= True)
        print('Writing climate table to disk...')
        df.to_csv( os.path.join(opath,ofilename),  index= False, compression= 'gzip' )
        print('Saved climate table in {}'.format(opath))
        
        #free memory up
        del dataset
        return None
    
    def get_target(self, ipath, opath, ifile_extension= 'ascii', ifilename= 'elsus_v2.asc' , ofilename= "target.csv.gz", save_plot= True):
        """
        Read the suceptibility map (raster) and convert it to a table.
        ipath: path of raster file
        opath: path where table (txt format) will be stored. 
        ifile_extension: 'ascii' or 'tif'. The input file needs to be raster anyways
        """
        print('Reading susceptibility map...')
        if (ifile_extension == 'ascii'):
            target = np.loadtxt(os.path.join(ipath, ifilename), skiprows= 6)
            target = target.astype(int) 
        if (ifile_extension == 'tif' ):
            target = rasterio.open( os.path.join(ipath, ifilename) ).read(1).astype(int) 
        
        if (save_plot):
            fig, ax = plt.subplots(1,1, figsize= (10, 10))
            ax.imshow(target, cmap= 'Reds_r')
            ax.set_xlabel('X [meters]', fontsize= 14)
            ax.set_ylabel('Y [meters]', fontsize= 14)
            ax.set_title('Susceptibility map', fontsize= 18)
            plt.savefig(os.path.join(opath,'target.png'), bbox_inches= 'tight')
            #plt.show()
        
        if (self.train):
            target=  self.confidence(target)
        
        print('target raster converted to numpy array.')
            
            
        #Create the table
        df = self.from_array_to_table(target)
        df.rename(columns={'value': 'target'}, inplace= True)
        print('Writing target table to disk ...')
        df.to_csv( os.path.join(opath,ofilename),  index= False, compression= 'gzip')
        print('Saved target table in {}'.format(opath))
        
        # free memory up
        del target
        return None
    
    def get_lithology_fraction_in_first_layer(self, ipath, opath, 
                                              ofilename='lithology_fraction_first_layer.csv.gz', 
                                              clay_file= 'CLYPPT_M_sl1_250m_clipped.tif', 
                                              silt_file= 'SLTPPT_M_sl1_250m_clipped.tif', 
                                              sand_file= 'SNDPPT_M_sl1_250m_clipped.tif'):
        """
        Function to obtain pct of clay, silt and sand 35 meters below surface.
        Inputs: clay_file: raster filename with clay pct 35 meters below surface.
                silt_file: raster filename with silt pct 35 meters below surface
                sand_file: raster filename with sand pct 35 meters below surface
                INPUT FILES SHOLD BE STORED IN THE SAME FOLER.
        Output: table stored in opath with the pct of clay, silt, sand.
        """
        # gather files in one list
        all_files = [clay_file, sand_file, silt_file]
        keywords= ['clay', 'sand', 'silt']

        fr_firstlayer= []

        for j in range(len(all_files)):
            print('Getting fraction of %s from first layer...'%keywords[j])
            first_layer = rasterio.open( os.path.join(ipath, all_files[j]) ).read(1).astype(int)
            
            if (self.train):
                first_layer= self.confidence(first_layer)
                
            # get table
            df = self.from_array_to_table(first_layer)
            df.rename(columns={'value': 'pct_{}_first_layer'.format(keywords[j])}, inplace= True)
            fr_firstlayer.append(df)
            # free up memory
            del first_layer
 
        #merge the tables
        print('Merging the tables..')
        df = pd.merge(fr_firstlayer[0], fr_firstlayer[1], on =['y', 'x'])
        df= pd.merge(df, fr_firstlayer[2], on= ['y', 'x'])
        print("Saving lithology first layer's table...")
        df.to_csv(os.path.join(opath, ofilename),  index= False, compression= 'gzip')
        print('Saved table in {}'.format(opath))
         
        del fr_firstlayer
        return None
    
     
    def get_field_capacity_table(self, ipath, opath, ofilename='field_capacity.csv.gz', clay_files=[], sand_files=[], silt_files=[] ):
        """
        Function that creates a table with field capacity.
        Inputs:
        clay/sand/silt_files: List of filenames of rasters of clay/sand/silt pct in all layers. 
                              Specify full path in the name.
        LISTS MUST HAVE THE SAME LENGTH
        """
        field_capacity_list= []

        for i in range(len(clay_files)): # runs over subsoil layers
            print('Computation of field capacity layer {}...'.format(i))
            fcpl = [] 
            claypct = rasterio.open(os.path.join(ipath,clay_files[i]) ).read(1).astype(int)
            siltpct = rasterio.open(os.path.join(ipath,silt_files[i]) ).read(1).astype(int)
            sandpct = rasterio.open(os.path.join(ipath,sand_files[i]) ).read(1).astype(int)
            
            if (self.train):
                # keep only points where confidence is good.
                claypct = self.confidence(claypct)
                siltpct = self.confidence(siltpct)
                sandpct = self.confidence(sandpct)

            matrix = [claypct, siltpct, sandpct]
            for k, subsoil in enumerate(['clay', 'silt', 'sand']):
                df= self.from_array_to_table(matrix[k])
                df.columns = ['y{}'.format(subsoil), 'x{}'.format(subsoil), 'p{}'.format(subsoil)]
                print(subsoil, df.shape)
                fcpl.append(df)
                
            fcpl= pd.concat(fcpl, axis=1)
            print(fcpl.shape )
            print(fcpl.columns)
            
            # Get field capacity
            fcpl['field_capacity_layer{}'.format(i)]= 'medium'
            fcpl.loc[(fcpl['pclay'] <=20) & (fcpl['psilt'] >=80) & (fcpl['psand'] >= 60), 'field_capacity']= 'low'
            fcpl.loc[(fcpl['pclay'] >=35) & (fcpl['psilt'] <=65) & (fcpl['psand'] >= 0), 'field_capacity']= 'high'

            # Append df to list to concatenate in future
            field_capacity_list.append( fcpl[['yclay', 'xclay', 'field_capacity_layer{}'.format(i)]] )
            # free up memory
            del claypct
            del siltpct
            del sandpct
        
        # Merge all the tables
        print('Field capacity computed for all lithology datasets.')
        fc = field_capacity_list[0][['yclay', 'xclay' ]].copy()

        for i in range(len(clay_files)):
            fc= pd.merge(fc, field_capacity_list[i], on = ['yclay', 'xclay' ], how= 'inner')

        fc.rename(columns= {'yclay': 'y', 'xclay':'x'}, inplace= True)
        
        # Write table 
        fc.to_csv(os.path.join(opath, ofilename), index= False, compression= 'gzip')
        print('Table of field capacity generated.')
        
        #freeup memory
        del field_capacity_list
        del fcpl
        return None
               

    def get_land_cover(self, ipath, opath,  ofilename='land_cover.csv.gz', files= {}, confidence_files= []):
        """
        Function to preprocess land cover images.
        Inputs:
            ipath: path of raw data
            opath: where to put the output
            ofilename: name of the output file
            files: (dict). dictionary with filenames of landcover.
            confidence_files: (dict). dictionary of filenames containing the confidence of files.
                             Default: {}
        Output: table with stored in opath with the land cover features.               
        """
        land_cover= []
        for i in files.keys():
            print('Reading {}...'.format(i))
            lc = rasterio.open(os.path.join(ipath, files[i]) ).read(1).astype(int) # convert matrix to int because values are codes.
            # Replace -9999 to 255
            lc =  np.where(lc != -9999 , lc, 255)
            lc =  np.where(lc != -999 , lc, 255)
            
            if (self.train):
                # keep only points where confidence is good.
                lc= self.confidence(lc)
                
            if (len(confidence_files)!=0):
                if (len(confidence_files) != len(files)):
                    raise ValueError('the number of confidence files is not the same as files.')
                # read confidence files
                conf= rasterio.open(os.path.join(ipath,confidence_files[i])).read(1)
                lc = np.where(conf>=80, lc, 255)
            
            # Get table
            df = self.from_array_to_table(lc)
            df.rename(columns= {'value': i}, inplace= True)
            land_cover.append(df)
            # free up memory
            del lc
        
        #Merge tables
        print('merging landcover tables...')
        if (len(land_cover)==1):
            landc = land_cover[0]
        if (len(land_cover)==2):
            landc = pd.merge(land_cover[0], land_cover[1], on=['y', 'x'], how= 'inner')
        if(len(land_cover)>2):
            landc = pd.merge(land_cover[0], land_cover[1], on=['y', 'x'], how= 'inner')
            for i in range(2, len(land_cover)):
                 landc = pd.merge(landc, land_cover[i], on=['y', 'x'], how= 'inner')
        
        # Write table 
        landc.to_csv(os.path.join(opath, ofilename), index= False, compression= 'gzip')
        print('Final table of land cover generated.')        
        
    def get_DEM_features(self, ipath, opath,  ofilename='dem.csv.gz', files= {}):
        """
        Function to preprocess land cover images.
        Inputs:
            ipath: path of raw data
            opath: where to put the output
            ofilename: name of the output file
            files: (dict). dictionary with filenames of dem features.
    
        Output: table stored in opath with DEM features.               
        """
        dem= []
        
        for i in files.keys():
            print('Reading {}...'.format(i))
            array = rasterio.open(os.path.join(ipath, files[i]) ).read(1).astype('float32') 
            # Replace -9999 to 255
            array =  np.where(array != -9999. , array, np.nan)
            array =  np.where(array != -999. , array, np.nan)
            
            if (self.train):
                # keep only points where confidence is good.
                array= self.confidence(array, dtype= np.float32) 
                array= np.where(array!= 255, array, np.nan)
            
            # Get table
            df = self.from_array_to_table(array, filter_nan= True)
            df.rename(columns= {'value': i}, inplace= True)
            dem.append(df)
            # free up memory
            del array
        
        #Merge tables
        print('merging DEM tables...')
        if (len(dem)==1):
            demt = dem[0]
        if (len(dem)==2):
            demt = pd.merge(dem[0], dem[1], on=['y', 'x'], how= 'inner')
        if(len(dem)>2):
            demt = pd.merge(dem[0], dem[1], on=['y', 'x'], how= 'inner')
            for i in range(2, len(dem)):
                 demt = pd.merge(demt, dem[i], on=['y', 'x'], how= 'inner')
        
        # Write table 
        demt.to_csv(os.path.join(opath, ofilename), index= False, compression= 'gzip')
        print('Final table of DEM generated.')        



def preprocessing_data():
    """
    Convert images into tables.
    For now, you need to set up manually the input and output paths
    """

    data_path= os.path.abspath( os.path.join('..', 'Data' ) ) 
    
    # input paths:
    climate_ipath= os.path.join(data_path, 'ELSUSv2', 'raw')
    target_ipath= os.path.join(data_path, 'ELSUSv2', 'raw')
    fraction1stlayer_ipath= os.path.join(data_path, 'lithology', 'raw')
    field_capacity_ipath= os.path.join(data_path, 'lithology', 'raw')
    landcover_ipath= os.path.join(data_path, 'land_cover', 'raw')
    dem_ipath= os.path.join(data_path, 'DEM_features', 'raw')
    
    #output paths:
    #Here I set my local drive, but in principle should be p drive
    #climate_opath= otpath= os.path.join(data_path, 'ELSUSv2', 'postprocessed')
    #target_opath= os.path.join(data_path, 'ELSUSv2', 'postprocessed')
    #fraction1stlayer_opath= os.path.join(data_path, 'lithology', 'postprocessed')
    #field_capacity_opath= os.path.join(data_path, 'lithology', 'postprocessed')
    #landcover_opath= os.path.join(data_path, 'land_cover', 'postprocessed')
    #dem_opath= os.path.join(data_path, 'DEM_features', 'postprocessed')
    opath= 'D:/landslides/' # I will copy the data from here to the commented paths
    
    #Setting up the instance. 
    cleaning= FromImageToTable(train= True)

    cleaning.get_climate(climate_ipath, opath) # climate data
    cleaning.get_target(target_ipath, opath, ifile_extension= 'ascii', save_plot= False) #target
    cleaning.get_lithology_fraction_in_first_layer(fraction1stlayer_ipath, opath)
    ##land cover
    cleaning.get_land_cover(landcover_ipath, opath,
                            files={'LCCS1_lc': 'LC_Prop1_2018_clipped.tif', 'LCCS2_lc': 'LC_Prop2_2018_clipped.tif', 'LCCS3_lc': 'LC_Prop3_2018_clipped.tif'},
                            confidence_files= {'LCCS1_lc': 'LC_Prop1_Assessment_2018_clipped.tif', 'LCCS2_lc': 'LC_Prop2_Assessment_2018_clipped.tif', 'LCCS3_lc': 'LC_Prop3_Assessment_2018_clipped.tif'}
                            )
    ##dem
    cleaning.get_DEM_features(dem_ipath, opath, files= {'tpi': 'europe_tpi.tif', 
                                                        'aspect': 'europe_aspect.tif',
                                                        'elevation': 'europe_elevation.tif',
                                                        'slope': 'europe_slope.tif'} )

    ##field capacity
    clay_ifiles = [f for f in os.listdir(field_capacity_ipath) if re.match('CLYPPT', f)]
    silt_ifiles = [f for f in os.listdir(field_capacity_ipath) if re.match('SLTPPT', f)]
    sand_ifiles = [f for f in os.listdir(field_capacity_ipath) if re.match('SNDPPT', f)]

    cleaning.get_field_capacity_table(field_capacity_ipath, 
                                           opath, 
                                           clay_files= clay_ifiles, 
                                           silt_files= silt_ifiles, 
                                           sand_files= sand_ifiles)
    
    
def data_enrichment():
    """
    Takes the tables obtained infunction preprocessing_data()
    and joins them.
    For now you need to set up manually the files
    """
    print('Reading files...')
    target= pd.read_csv('D:/landslides/target.csv.gz')
    climate= pd.read_csv('D:/landslides/elsus_climate.csv.gz' )
    lfrac= pd.read_csv('D:/landslides/lithology_fraction_first_layer.csv.gz')
    fieldc= pd.read_csv('D:/landslides/field_capacity.csv.gz')
    dem= pd.read_csv('D:/landslides/dem.csv.gz')
    landc= pd.read_csv('D:/landslides/land_cover.csv.gz')
    
    # concat tables
    tables= [climate, lfrac, fieldc, dem, landc]
    
    # Start the join with the target. Do an inner join.
    print('Starting data enrichment process...')
    df1 = target.copy()
    for i in range(len(tables)):
        df1 = pd.merge(df1, tables[i], on=['y', 'x'], how= 'inner')
        print(df1.shape)
    
    print('Process finished.')
    return df1



#if __name__ == '__main__':
    #preprocessing_data()
    data = data_enrichment()
    data.to_csv('D:/landslides/Landslide_good_confidence_new_features_final.csv.gz')
    #os.system('shutdown -s')

    

    
    
    

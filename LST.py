'''This script adapts functions genearted for Pendergrass and Knutti (2018) to evaluate the rd50 distributions
for classification of large storm dominance within CMIP6 GCMs of high and low resolution'''

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Given one year of precip data, calculate the number of days for half of precipitation
# Ignore years with zero precip (by setting them to NaN).
# Ignore years with more than 30% missing data
def oneyear(thisyear):

    missingthresh = 0.1 # threshold of missing data fraction at which a year is thrown out
    # thisyear is one year of data, (an np array) with the time variable in the leftmost dimension
    dims=thisyear.shape
    nd=dims[0]
    missingfrac = (np.sum(np.isnan(thisyear),axis=0)/nd)
    ptot=np.sum(thisyear,axis=0)
    sortandflip=-np.sort(-thisyear,axis=0)
    cum_sum=np.cumsum(sortandflip,axis=0)
    ptotnp=np.array(ptot)
    ptotnp[np.where(ptotnp == 0)]=np.nan
    pfrac = cum_sum / np.tile(ptotnp[np.newaxis,:,:],[nd,1,1])
    ndhy = np.full((dims[1],dims[2]),np.nan)
    x=np.linspace(0,nd,num=nd+1,endpoint=True)
    z=np.array([0.0])
    for ij in range(dims[1]):
        for ik in range(dims[2]):
            p=pfrac[:,ij,ik]
            y=np.concatenate([z,p])
            ndh=np.interp(0.5,y,x)
            ndhy[ij,ik]=ndh
    ndhy[np.where(missingfrac > missingthresh)] = np.nan
    return ndhy

def convert_lons_to_norm(ds):
    '''Convert the longitues to -180 to 180 if they are 0 to 360'''
    # Check to convert the nc to -180-180 lon coords
    if max(ds.lon.values) > 180:
        ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180
        ds = ds.sortby(['lat','lon'])

    return ds


def annual_rd50_wrapper(allpaths,
                        outpath):
    '''Takes a list of paths of daily GCM precipitation data and an output path for where to save the
    calculation of annual rd50'''
    for path in allpaths:
        ds = xr.open_dataset(path)
        ds = convert_lons_to_norm(ds)
        if ds.indexes['time'].dtype != 'datetime64[ns]':
                ds['time'] = ds.indexes['time'].to_datetimeindex()

        yearlist = np.unique(ds.time.dt.year)
        print(yearlist)

        years=[]
        for year in yearlist:
            years.append(str(year))
        #
        ny=len(years)
        #
        cfy = np.full((ny,ds.lon.size,ds.lat.size),np.nan)
        #
        ds.load()
        for year in range(ny):

            tempds = ds.where(ds.time.dt.year==yearlist[year],drop=True)
            # print(tempds)
            thisyear=tempds.pr
            thisyearnp=np.array(thisyear.transpose('time','lon','lat'))
            ndhy=oneyear(thisyearnp)
            cfy[year,:,:]=ndhy

        print(cfy.shape)
        allyears_rd50 = xr.DataArray(cfy,dims=['year','lon','lat'],coords={'year':yearlist,
                                                                           'lon':ds.lon,
                                                                           'lat':ds.lat})
        allyear_outpath = outpath+'/'+os.path.basename(path)
        allyears_rd50.to_netcdf(allyear_outpath)
        print(allyears_rd50)


if __name__=='__main__':

    # Path to coarse resolution and high resoution GCMs
    cmip6_lrdirpath = '/Volumes/testpath'

    cmip6_hrdirpath = '/Volumes/testpath'

    # Gather all netcdf files within the directories
    import glob
    hrpathlist = glob.glob(cmip6_hrdirpath+'/*.nc')
    allpaths = glob.glob(cmip6_lrdirpath+'/*.nc')
    # Join the path lists togther
    allpaths.extend(hrpathlist)
    print(len(allpaths))

    # Set an outpath
    outpath = '/Volumes/outpath'

    annual_rd50_wrapper(allpaths,
                        outpath)

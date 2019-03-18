---
layout: post
title: "使用ecCodes读取ECMWF模式输出的GRIB文件"
tags: [气象]
date: 2018-09-29
---
欧洲中心的GRIB数据文件的编解码将使用[ecCodes](https://confluence.ecmwf.int//display/ECC/ecCodes+Home)来进行

本文主要介绍如何使用ecCodes来读取欧洲中心细网格模式输出的GRIB文件中的变量（地表和气压层）

**说明**
+ ecCodes最简单的的安装方法是使用Anaconda(conda install -c conda-forge python-eccodes)
+ 在从文件读取变量之前最好使用grib_dump命令对文件内容有一个大体的了解

**代码**
```python
# -*- coding: UTF-8 -*-
import sys
import eccodes
import numpy as np
from datetime import datetime, timedelta

def getVar(file, varname):
    ### file表示要读取的GRIB文件(字符串类型)
    ### varname表示要读取的变量名(字符串类型)     
    var_summary = dict() #存储变量信息   

    iid = eccodes.codes_index_new_from_file(file, keys=['shortName'])
    vars_all = eccodes.codes_index_get(iid, key='shortName') #all variables in the file
    if varname not in vars_all:
        print("Error: {0} is not in the list of variables")
        sys.exit()
    var_summary['shortname'] = varname

    eccodes.codes_index_select(iid, key='shortName', value=varname)
    gid = eccodes.codes_new_from_index(iid)

    ### variable type
    try:
        level_type = 'sfc' if varname=='ptype' else eccodes.codes_get(gid, 'indicatorOfTypeOfLevel')
        var_summary['type'] = 'surface' if level_type=='sfc' else 'pressure levels(hPa)'
    except:
        print('The function only read the surface or pressure level variables, {0} is model level variable' \
              .format(varname))
        sys.exit()

    ### time info
    ymdh_start_utc = str(eccodes.codes_get(gid, 'dataDate')) + '%02d'%(eccodes.codes_get(gid, 'dataTime')//100)
    delta_hours = int(eccodes.codes_get(gid, 'endStep'))

    ymdh_start_bj_date = datetime.strptime(ymdh_start_utc,'%Y%m%d%H') + timedelta(hours=8) #transform from utc to beijing
    ymdh_pred_bj_date = ymdh_start_bj_date + timedelta(hours=delta_hours)

    ymdh_start_bj = ymdh_start_bj_date.strftime('%Y%m%d%H')
    ymdh_pred_bj = ymdh_pred_bj_date.strftime('%Y%m%d%H')
    var_summary['start_time'] = ymdh_start_bj
    var_summary['predict_time'] = ymdh_pred_bj

    ### long name and unit
    long_name = eccodes.codes_get(gid, 'name')
    unit = eccodes.codes_get(gid, 'units')
    var_summary['longname'] = long_name
    var_summary['unit'] = unit

    ### lat/lon info
    lat_ec_start = eccodes.codes_get(gid, 'latitudeOfFirstGridPointInDegrees')
    lat_ec_end = eccodes.codes_get(gid, 'latitudeOfLastGridPointInDegrees')
    lat_ec_delta = eccodes.codes_get(gid, 'jDirectionIncrementInDegrees')
    s_to_n = False if eccodes.codes_get(gid, 'jScansPositively')==0 else True

    lon_ec_start = eccodes.codes_get(gid, 'longitudeOfFirstGridPointInDegrees')
    lon_ec_end = eccodes.codes_get(gid, 'longitudeOfLastGridPointInDegrees')
    lon_ec_delta = eccodes.codes_get(gid, 'iDirectionIncrementInDegrees')
    w_to_e = False if eccodes.codes_get(gid, 'iScansNegatively')==1 else True
    lon_ec_start = 360+lon_ec_start if lon_ec_start<0 else lon_ec_start #change to the range 0-360
    lon_ec_end = 360+lon_ec_end if lon_ec_end<0 else lon_ec_end #change to the range 0-360

    var_summary['lat_start'] = lat_ec_start  
    var_summary['lat_end'] = lat_ec_end
    var_summary['lat_delta'] = lat_ec_delta
    var_summary['south_to_north'] = s_to_n
    var_summary['lon_start'] = lon_ec_start  
    var_summary['lon_end'] = lon_ec_end
    var_summary['lon_delta'] = lon_ec_delta
    var_summary['west_to_east'] = w_to_e

    ### levels and data values
    levels = []
    data_levels = []
    numlat = eccodes.codes_get(gid, 'Nj')
    numlon = eccodes.codes_get(gid, 'Ni')
    while gid:
        level = eccodes.codes_get(gid, 'level')
        data_ec = eccodes.codes_get_values(gid).reshape(numlat, numlon)
        levels.append(level)
        data_levels.append(data_ec)
        eccodes.codes_release(gid)
        gid = eccodes.codes_new_from_index(iid)
    eccodes.codes_index_release(iid)

    if level_type=='pl': var_summary['levels']=sorted(levels)
    var_summary['values'] = np.array(data_levels)[np.argsort(levels)]

    return var_summary
```

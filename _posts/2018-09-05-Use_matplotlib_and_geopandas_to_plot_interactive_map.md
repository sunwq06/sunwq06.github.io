---
layout: post
title: "使用matplotlib和geopandas画交互式地图"
tags: [气象]
date: 2018-09-05
---

+ **目标：** 画出山东省日最高（低）温的预报偏差空间分布并进行一定程度的交互
+ **数据：**
  + 地图数据（.shp）
  + 站点经纬度数据（.txt）
  + 各站点一段时间以来温度预报的平均偏差数据（.txt）
  + [数据下载](https://pan.baidu.com/s/1_rBQ44aC4qwzG_7NLxOlxw)
+ **代码功能：**
     + 画出山东省地图及省内气象观测站的位置
     + 将各市的日最高（低）温预报偏差（站点平均）按从小到大排序，颜色越深表示预报偏差越大
     + 地图中的点越大表示在该站点的预报偏差越大，青色的点表示预报偏差排在后30%的站点
     + 鼠标右键点击城市区域，打印城市名称、排位以及对应的预报偏差
     + 鼠标左键点击站点，打印站点信息以及对应的预报偏差
+ **代码步骤：**
     1. 导入相应的包
 ```python
 import shapely
 import numpy as np
 import pandas as pd
 import geopandas as gpd
 import matplotlib.pyplot as plt
 from matplotlib.patches import Polygon
 from matplotlib.colors import BoundaryNorm
 from matplotlib.colorbar import ColorbarBase
 from matplotlib.collections import PatchCollection
 from matplotlib.collections import PathCollection
 ### 使图片中正常显示中文
 plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
 plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
 ```

     2. 获取站点信息以及各站点预报的平均偏差，将各市内的站点偏差平均得到各市预报的平均偏差
 ```python
 ### 获取站点信息
 '''
 文件示例：
 station  lat lon height  city
 54709 37.233  116.067 24.8  德州市
 '''
 location = pd.read_csv('nation_stations_sd.txt',sep='\t',encoding="gb2312").sort_values(by='station')
 stations = [str(s) for s in location['station']]
 lats = [round(x,3) for x in location['lat']]
 lons = [round(x,3) for x in location['lon']]
 heights = [round(x,2) for x in location['height']]
 ### 获取各站点预报的平均偏差
 '''
 文件示例：
 station bias_mos
 54709 0.839
 '''
 Ttype = 'min' #or 'max'
 clock = '08' #or '20'
 Ttype_c = {'min':'最低', 'max':'最高'} #for plot
 bias = pd.read_csv('./daily_{0}_{1}'.format(Ttype, clock), sep='\t').sort_values(by='station')
 station_val = [round(v,2) for v in bias['bias_mos']]
 ### 获取各市预报的平均偏差并按偏差从小到大排序
 city_val_df = pd.merge(bias[['station','bias_mos']], location[['station','city']], on='station') \
                         .groupby('city')['bias_mos'].mean().reset_index().sort_values(by='bias_mos')
 city_val_df = city_val_df.assign(order=range(1, city_val_df.shape[0]+1)).set_index('city')
 ```

     3. 使用geopandas读取地图文件
```python
### 使用geopandas读取山东省地图文件
sd = gpd.read_file('./中国地图/province/sd_city.shp')
### 将地图文件的geometry属性转化为Polygon对象并关联相关信息
patches = []
values = []
orders = []
cities = []
for poly,city in zip(sd.geometry, sd.NAME99):
            if isinstance(poly, shapely.geometry.polygon.Polygon):
                cities.append(city)
                patches.append(Polygon(np.asarray(poly.exterior)))      
                values.append(city_val_df.loc[city]['bias_mos'])
                orders.append(city_val_df.loc[city]['order']-0.5) #for plot
            if isinstance(poly, shapely.geometry.multipolygon.MultiPolygon):
                cities += [city for p in poly]
                patches += [Polygon(np.asarray(p.exterior)) for p in poly]
                values += [city_val_df.loc[city]['bias_mos'] for p in poly]
                orders += [city_val_df.loc[city]['order']-0.5 for p in poly] #for plot
patches = PatchCollection(patches)
patches.set_array(np.array(orders))
patches.set_picker(True)
```

     4. 画山东地图并按各市预报偏差的排序填色
```python
### 画山东省地图
fig, ax = plt.subplots(figsize=(8,6))
ax.set_title('{0}时起报的24h{1}温'.format(clock, Ttype_c[Ttype]), fontsize=18)
ax.add_collection(patches, autolim=True)
ax.autoscale_view()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.spines['top'].set_visible(False)  #去掉上边框  
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
cityname_loc = {u'东营':(118.55, 37.58), u'滨州':(117.84, 37.55), u'淄博':(117.97, 36.56), \
                        u'莱芜':(117.68, 36.23), u'潍坊':(119.04, 36.58), u'泰安':(117.04, 36.00), \
                        u'菏泽':(115.69, 35.20), u'济宁':(116.51, 35.24), u'枣庄':(117.33, 34.95), \
                        u'临沂':(118.24, 35.25), u'日照':(119.13, 35.57), u'青岛':(120.30, 36.48), \
                        u'烟台':(120.87, 37.14), u'威海':(121.99, 37.10), u'济南':(117.27, 36.73), \
                        u'聊城':(115.77, 36.48), u'德州':(116.55, 37.14)}
for cityname, (lon, lat) in cityname_loc.items():
            ax.text(lon, lat, cityname, color='y', ha='center', size='xx-large')
### 设置colorbar信息
cmap = plt.cm.Reds
bounds = np.linspace(0, city_val_df.shape[0], city_val_df.shape[0]+1)
norm = BoundaryNorm(bounds, cmap.N)
ticks = np.linspace(1, city_val_df.shape[0], city_val_df.shape[0])-0.5
ticklabels = [str(int(t)) for t in np.linspace(1, city_val_df.shape[0], city_val_df.shape[0])]
### 创建colorbar
ax2 = fig.add_axes([0.9, 0.1, 0.03, 0.8])
ax2.set_title('Order')
cb = ColorbarBase(ax2, cmap=cmap, norm=norm, ticks=ticks)
cb.set_ticklabels(ticklabels)
patches.set_cmap(cmap)
patches.set_norm(norm)
```

     5. 在地图上画出观测站点的散点图（点的大小对应站点预报偏差大小，青色的点表示偏差排在后30%的站点）
```python
### 画观测站点的散点图
smin = 10
smax = 100
interval = np.max(station_val)-np.min(station_val)
sizes = smin+(smax-smin)*(station_val-np.min(station_val))/interval
colors = ['Blue' if val<np.percentile(bias['bias_mos'], 70) else 'Aqua' for val in station_val]
ax.scatter(lons, lats, picker=1, s=sizes, c=colors)
```

     6. 在图中添加交互功能
```python
### 在图形中添加交互
def onpick(event):
            this = event.artist
            ind = event.ind[0]
            button = event.mouseevent.button
            if button==1 and isinstance(this, PathCollection):
                print('站号: {0}  纬度: {1}  经度: {2}  高度: {3}  所属城市: {4}  预报偏差: {5}' \
                      .format(stations[ind], lats[ind], lons[ind], heights[ind], \
                              location.iloc[ind]['city'], station_val[ind]))
            elif button==3 and isinstance(this, PatchCollection):
                print('城市: {0}  排位: {1}  预报偏差: {2}'.format(cities[ind], int(orders[ind]+0.5), \
                                                                   round(values[ind],2)))  
            return
fig.canvas.mpl_connect('pick_event', onpick)
```


+ **生成地图示例：**

  ![img](/img/sd_temp.png)

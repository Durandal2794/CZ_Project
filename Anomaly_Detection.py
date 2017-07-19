
# coding: utf-8

# In[9]:

import os
import sys
import re
import numpy as np
import pandas as pd
import scipy

import plotly.plotly as py 
from plotly.graph_objs import *
import plotly.graph_objs as go
from plotly import __version__
import plotly.offline as offline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

from pyculiarity import detect_ts

import rpy2 as r
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import conversion
from rpy2.robjects import pandas2ri
from IPython.display import display, HTML, IFrame

R = ro.r
# pandas2ri.activate()
# plotly = importr("plotly")
# d3heatmap = importr("d3heatmap")
# forcats = importr("forcats")
# anomaly = importr("AnomalyDetection")

root_dir = os.getcwd()
output_dir = root_dir


# In[10]:

iplot([{"x": [1, 2, 3], "y": [3, 1, 6]}])


# In[11]:

x = np.random.randn(2000)
y = np.random.randn(2000)
iplot([Histogram2dContour(x=x, y=y, contours=Contours(coloring='heatmap')),
       Scatter(x=x, y=y, mode='markers', marker=Marker(color='white', size=3, opacity=0.3))], show_link=False)


# In[12]:

iplot(cf.datagen.lines().iplot(asFigure=True,
                               kind='scatter',xTitle='Dates',yTitle='Returns',title='Returns'))


# In[ ]:




# In[26]:

iplot([
    {
    "x": np.arange(0,50,0.1),
    "y": [np.sin(i) for i in np.arange(0,50,0.1)]
    }
])


# In[32]:

x = np.arange(0,50,0.1)
y = [np.sin(i) for i in np.arange(0,50,0.1)]


# In[112]:

y[50] = 2
y[100] = -8
y[150] = 4
y[200] = 3
y[250] = -4
y[300] = -3
y[350] = 6
y[400] = 7


# In[113]:

iplot([{"x": x, "y": y}])


# In[114]:

t = pd.date_range(start = "January 1, 2017", end = "December 31, 2017", freq = "D")
v = y[0:365]
df = pd.DataFrame(v,t)
df.reset_index(inplace = True)
df.columns = ["Time","Value"]
df.head()


# In[115]:

results = detect_ts(df, max_anoms=0.02, direction = "both")


# In[117]:

results


# In[118]:

results["anoms"]


# In[ ]:





# coding: utf-8

# # Scrapalyze:  Web Scraper and Sentiment Analyzer
# Authored by Chris Cotton, 07/17/2017
# 
# chris.j.cotton@me.com
# 
# 
# ## Purpose:
# 
# This notebook leverages the Goose and Textblob modules for Python which enable the scraping, parsing, and analyzing of sentiment and subjectivity of natural language data on the internet.  Goose scrapes and cleans raw web data; Textblob leverages the NLTK module for Python to perform sentiment and subjectivity analysis on the data.
# 
# Some bubble charts are displayed for fun, just plotting sentiment vs. subjectivity of various parts of the data scraped (title, metadata, text, etc.).  The heatmaps display all of the attributes of the text analyzed (6 of them) vs. all observations (websites) in the data set.  A bottom-up, hierarchical, agglomerative clustering algorithm (Ward clustering) is applied whose objective function is to find pairs of rows with the most similar variance (smallest difference in erorr sum of squares between rows).  A taxonomy is created that can be pruned at any level to find clades of websites with similar sentiments.  The same algorithm is applied column-wise, to find attributes that are, holistically, most similar to one another *across* the websites.
# 
# 
# ## Inputs:
# 
# The user defines a list of domains, and the program tries to scrape and analyze every domain in the list that it can.
# 
# 
# ## Execution:
# 
# After defining your list, click the "Cell" menu, and click "Run All."

# In[1]:

from __future__ import division
import os
import sys
import re
import numpy as np
import pandas as pd
import scipy

import nltk, re, pprint
from nltk import word_tokenize
import urllib2 as ul2
from goose import Goose
from textblob import TextBlob

import plotly.plotly as py 
from plotly.graph_objs import *
import plotly.graph_objs as go
from plotly import __version__
import plotly.offline as offline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import rpy2 as r
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import conversion
from rpy2.robjects import pandas2ri
from IPython.display import display, HTML, IFrame


# In[2]:

get_ipython().magic(u'load_ext rpy2.ipython')

import warnings
warnings.filterwarnings('ignore')


# In[3]:

get_ipython().magic(u'R require(ggplot2); require(tidyr); require(plotly); require(d3heatmap)')


# In[4]:

R = ro.r
pandas2ri.activate()
# plotly = importr("plotly")
# d3heatmap = importr("d3heatmap")
# forcats = importr("forcats")
# anomaly = importr("AnomalyDetection")

root_dir = os.getcwd()
output_dir = root_dir


# In[5]:

g = Goose()


# In[6]:

domain_list = ["google.com","youtube.com","facebook.com","baidu.com","wikipedia.org","yahoo.com","reddit.com","google.co.in","qq.com","amazon.com","taobao.com","twitter.com","tmall.com","google.co.jp","vk.com","live.com","sohu.com","instagram.com","sina.com.cn","jd.com","weibo.com","360.cn","google.de","google.co.uk","google.com.br","list.tmall.com","linkedin.com","google.fr","google.ru","yandex.ru","netflix.com","google.com.hk","yahoo.co.jp","google.it","ebay.com","t.co","pornhub.com","google.es","imgur.com","bing.com","twitch.tv","msn.com","onclkds.com","gmw.cn","tumblr.com","google.com.mx","google.ca","alipay.com","xvideos.com","livejasmin.com","mail.ru","ok.ru","microsoft.com","aliexpress.com","wordpress.com","hao123.com","stackoverflow.com","imdb.com","amazon.co.jp","github.com","blogspot.com","csdn.net","wikia.com","pinterest.com","apple.com","google.com.tr","popads.net","youth.cn","bongacams.com","office.com","paypal.com","google.com.tw","google.com.au","whatsapp.com","microsoftonline.com","google.pl","xhamster.com","detail.tmall.com","diply.com","google.co.id","adobe.com","nicovideo.jp","craigslist.org","amazon.de","txxx.com","amazon.in","google.com.ar","porn555.com","coccoc.com","dropbox.com","booking.com","thepiratebay.org","google.com.pk","googleusercontent.com","google.co.th","pixnet.net","china.com","google.com.eg","soso.com","bbc.co.uk","tianya.cn","google.com.sa","amazon.co.uk","savefrom.net","fc2.com","bbc.com","rakuten.co.jp","uptodown.com","so.com","soundcloud.com","google.com.ua","mozilla.org","xnxx.com","cnn.com","amazonaws.com","quora.com","ask.com","google.nl","ettoday.net","nytimes.com","naver.com","adf.ly","dailymotion.com","clicksgear.com","google.co.za","steamcommunity.com","onlinesbi.com","google.co.ve","espn.com","google.co.kr","salesforce.com","chase.com","fbcdn.net","blogger.com","stackexchange.com","ebay.de","vice.com","vimeo.com","theguardian.com","chaturbate.com","steampowered.com","blastingnews.com","ebay.co.uk","mediafire.com","tribunnews.com","indeed.com","buzzfeed.com","openload.co","google.gr","avito.ru"]


# In[7]:

url_list = ["http://www." + i for i in domain_list]


# In[8]:

def parse_and_analyze(url):
    domain = url.replace("http://www.","")
    parsed = g.extract(url)

    title = parsed.title
    meta = parsed.meta_description
    text = parsed.cleaned_text
    overall = title + " " + meta + " " + text

    title_blob = TextBlob(title)
    meta_blob = TextBlob(meta)
    text_blob = TextBlob(text)
    overall_blob = TextBlob(overall)

    title_sentiment = title_blob.sentiment.polarity
    meta_sentiment = meta_blob.sentiment.polarity
    text_sentiment = text_blob.sentiment.polarity
    overall_sentiment = overall_blob.sentiment.polarity

    title_subjectivity = title_blob.sentiment.subjectivity
    meta_subjectivity = meta_blob.sentiment.subjectivity
    text_subjectivity = text_blob.sentiment.subjectivity
    overall_subjectivity = overall_blob.sentiment.subjectivity

    results_list = [
    domain, title, meta, text,
    title_sentiment, meta_sentiment, text_sentiment, overall_sentiment,
    title_subjectivity, meta_subjectivity, text_subjectivity, overall_subjectivity
    ]

    return results_list



def merge(url_list):
    results_list = []
    successful = []
    failed = []

    i = 0
    s = 0
    f = 0

    for url in url_list:
        try:
            results = parse_and_analyze(url)
            results_list.append(results)
            successful.append(url)
            i += 1
            s += 1
            print url + " successful!"
            print "{} urls processed so far; {} successful; {} failed.".format(i, s, f)
        except:
            failed.append(url)
            i += 1
            f += 1
            print url + " failed..."
            print "{} urls processed so far; {} successful; {} failed.".format(i, s, f)

    df = pd.DataFrame(results_list)

    return df, successful, failed


# In[9]:

df, successful, failed = merge(url_list)


# In[10]:

df.columns = [
"Domain", "Title", "Meta", "Text",
"Title Sentiment", "Meta Sentiment", "Text Sentiment", "Overall Sentiment",
"Title Subjectivity", "Meta Subjectivity", "Text Subjectivity", "Overall Subjectivity",
]


# In[11]:

df.to_csv("/Users/chrcotto/cloudzero_scrapy/cloudzero_results.txt", sep = "\t", encoding = "utf-8")


# In[12]:

df


# In[13]:

df.describe()


# In[14]:

df[df.sum(axis = 1) > 0].describe()


# In[15]:

hover_text = []

for index, row in df.iterrows():
    hover_text.append(
        ('Domain: {domain}<br>'+
        'Title Sentiment: {title}<br>'+
        'Text Sentiment: {text}<br>').format(
                                        domain = row["Domain"],
                                        title = row["Title Sentiment"],
                                        text = row["Text Sentiment"]
                                        )
                     )


df["Hover Text"] = hover_text


trace0 = go.Scatter(
    x = df["Title Sentiment"],
    y = df["Text Sentiment"],
    text = df["Hover Text"],
    mode = "markers",
    marker = dict(
        size=[40] * len(df),
    )
)


layout = go.Layout(
    title = "Text vs. Title Sentiment for Top Alexa Sites",
    xaxis = dict(
        title = "Title Sentiment Score",
        gridcolor = "rgb(255, 255, 255)",
        zerolinewidth = 1,
        ticklen = 5,
        gridwidth = 2,
    ),
    yaxis=dict(
        title = "Text Sentiment Score",
        gridcolor = "rgb(255, 255, 255)",
        zerolinewidth = 1,
        ticklen = 5,
        gridwidth = 2,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)


data = [trace0]

fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename = "bubblechart-size")


# In[16]:

hover_text = []

for index, row in df.iterrows():
    hover_text.append(
        ('Domain: {domain}<br>'+
        'Subjectivity: {subjectivity}<br>'+
        'Sentiment: {sentiment}<br>').format(
                                        domain = row["Domain"],
                                        subjectivity = row["Text Subjectivity"],
                                        sentiment = row["Text Sentiment"]
                                        )
                     )


df["Hover Text2"] = hover_text


trace0 = go.Scatter(
    x = df["Text Subjectivity"],
    y = df["Text Sentiment"],
    text = df["Hover Text2"],
    mode = "markers",
    marker = dict(
        size=[40] * len(df),
    )
)


layout = go.Layout(
    title = "Text Sentiment vs. Subjectivity for Top Alexa Sites",
    xaxis = dict(
        title = "Text Subjectivity Score",
        gridcolor = "rgb(255, 255, 255)",
        zerolinewidth = 1,
        ticklen = 5,
        gridwidth = 2,
    ),
    yaxis=dict(
        title = "Text Sentiment Score",
        gridcolor = "rgb(255, 255, 255)",
        zerolinewidth = 1,
        ticklen = 5,
        gridwidth = 2,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)


data = [trace0]

fig = go.Figure(data = data, layout = layout)
py.iplot(fig, filename = "bubblechart-size")


# In[17]:

df_heatmap = df[["Domain","Title Sentiment","Meta Sentiment","Text Sentiment",
                "Title Subjectivity","Meta Subjectivity","Text Subjectivity"]]

df_heatmap.set_index("Domain", inplace = True)
df_heatmap = df_heatmap[df_heatmap.sum(axis = 1) > 0]
df_row_normalized = df_heatmap.div(df_heatmap.sum(axis = 1), axis = 0)
df_col_normalized = df_heatmap.div(df_heatmap.sum(axis = 0), axis = 1)


# In[18]:

df_heatmap.head()


# In[19]:

df_row_normalized.head()


# In[20]:

df_col_normalized.head()


# In[21]:

get_ipython().run_cell_magic(u'R', u'-i df_heatmap', u'p <- d3heatmap(df_heatmap, colors = "YlGnBu", theme = "dark", height = 800, width = 800,\n          k_row = 18, k_col = 6, scale = "none", symm = TRUE,\n            hclustfun = function(x) hclust(x, method = "ward.D2"),\n                na.rm = TRUE, xaxis_font_size = 12, yaxis_font_size = 11)\nhtmlwidgets::saveWidget(as.widget(p), "/users/chrcotto/df_heatmap.html", selfcontained = T)')


# In[22]:

IFrame("df_heatmap.html", width = 900, height = 900)


# In[23]:

get_ipython().run_cell_magic(u'R', u'-i df_row_normalized', u'p2 <- d3heatmap(df_row_normalized, colors = "YlGnBu", theme = "dark", height = 800, width = 800,\n          k_row = 18, k_col = 6, scale = "row", symm = TRUE,\n            hclustfun = function(x) hclust(x, method = "ward.D2"),\n                na.rm = TRUE, xaxis_font_size = 12, yaxis_font_size = 11)\nhtmlwidgets::saveWidget(as.widget(p2), "/users/chrcotto/df_heatmap_row_normal.html", selfcontained = T)')


# In[24]:

IFrame("df_heatmap_row_normal.html", width = 900, height = 900)


# In[25]:

get_ipython().run_cell_magic(u'R', u'-i df_col_normalized', u'p3 <- d3heatmap(df_col_normalized, colors = "YlGnBu", theme = "dark", height = 800, width = 800,\n          k_row = 18, k_col = 6, scale = "col", symm = TRUE,\n            hclustfun = function(x) hclust(x, method = "ward.D2"),\n                na.rm = TRUE, xaxis_font_size = 12, yaxis_font_size = 11)\nhtmlwidgets::saveWidget(as.widget(p3), "/users/chrcotto/df_heatmap_col_normal.html", selfcontained = T)')


# In[26]:

IFrame("df_heatmap_col_normal.html", width = 900, height = 900)


# In[ ]:




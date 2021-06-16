---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.10.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# WSB Forecasting

The purpose of this notebook is to outline the process of gathering and processing data on individual stock mentions and sentiment throughout Wallstreetbets. From there, some basic forecasting will be done on stock mentions.

For consistency, some sample data is included in this notebook, it is real data from the database, including posts and comments from the last week (so June 6 - June 13) and the list of all authors registered in the database so far. The sample data is saved in a json format in order to prevent issues with delimiters. Since Reddit is an online forum, who knows what kinds of characters they use, no delimiter is safe.

```python
import pandas as pd
import pickle

posts = pd.read_json(open('./posts.json','r',encoding = 'cp850'),orient='index')
comments = pd.read_json(open('./comments.json','r',encoding='cp850'),orient='index')
```

```python
posts.head()
```

```python
comments.head()
```

Those both loaded in correctly so looks like we're good to move on to the processing steps (segmenting by date and counting up stock mentions)

```python
#Mapping the new lines into regular spaces so that there's always a space between 
#words
comments['text'] = comments['text'].map(lambda x: x.replace('\n',' ').lower())
posts['post_title'] = posts['post_title'].map(lambda x: x.replace('\n',' ').lower())
posts['self_text'] = posts['self_text'].map(lambda x: x.replace('\n',' ').lower())

```

```python
#Imports pre-generated list converting stock tickers to their names and vice versa
#as well as some stopwords

nameToTicker = pickle.load(open('./nameToTicker.p','rb'))
tickerToName = pickle.load(open('./tickerToName.p','rb'))
stopwords = pickle.load(open('./stopwords.p','rb'))
```

```python
posts['date_posted'].map(lambda x: x[0:10]).unique()
```

```python

```

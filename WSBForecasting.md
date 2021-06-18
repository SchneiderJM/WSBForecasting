---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.1
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
#Gathering the dates in the dataset
dates = posts['date_posted'].map(lambda x: x[0:10]).unique()

#Counting up mentions of stocks by date
stock_mentions = {date:{ticker.lower():0 for ticker in tickerToName.keys() if (ticker==ticker)} for date in dates}
```

```python
#Finds if a given ticker is in the text
#Since tickers are short and often appear in common words, just counting up their direct appearances
#would not be very accurate
#The function below checks a few extra cases that would be hard to represent in just a lambda function
def find_ticker_in_text(text,ticker,stopwords):
    #Some tickers are used as words, so this removes common words
    #Luckily, stopwords tend to be short words that are also tickers, so typical NLP stopwords
    #Are a decent proxy
    if (ticker not in stopwords):
        #Handles if the ticker is mentioned mid sentence or at the start of a sentence
        if (' '+ticker+' ' in text):
            return(True)
        #Handles if the ticker is at the end of the sentence
        elif (' '+ticker+'.' in text):
            return(True)
        #Handles if the ticker is at the start of the text
        elif (text[0:(len(ticker)+1)] == (ticker+' ')):
            return(True)
        #Handles if the ticker is at the end of the text
        elif (text[-1*(len(ticker)-2):-1] == (' '+ticker)):
            return(True)
        else:
            return(False)
    else:
        return(False)
```

```python
for date in dates:
    #Comments made on the date being examined
    dated_comments = comments[comments['date_posted'].map(lambda x: True if date in x else False)]
    #Posts made on the date being examined
    dated_posts = posts[posts['date_posted'].map(lambda x: True if date in x else False)]
    
    #Counts up ticker mentions
    for ticker in stock_mentions[date].keys():
        stock_mentions[date][ticker] += sum(dated_comments['text'].map(lambda x: find_ticker_in_text(x,ticker,stopwords)))
        stock_mentions[date][ticker] += sum(dated_posts['post_title'].map(lambda x: find_ticker_in_text(x,ticker,stopwords)))
        stock_mentions[date][ticker] += sum(dated_posts['self_text'].map(lambda x: find_ticker_in_text(x,ticker,stopwords)))
        
    #Counts up stock name mentions
    for name in nameToTicker.keys():
        stock_mentions[date][ticker] += sum(dated_comments['text'].map(lambda x: find_ticker_in_text(x,name,stopwords)))
        stock_mentions[date][ticker] += sum(dated_posts['post_title'].map(lambda x: find_ticker_in_text(x,name,stopwords)))
        stock_mentions[date][ticker] += sum(dated_posts['self_text'].map(lambda x: find_ticker_in_text(x,name,stopwords)))
```

```python
stock_mentions['2021-06-11']['amc']
```

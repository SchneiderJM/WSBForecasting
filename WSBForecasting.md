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

<!-- #raw -->
#Some processing that has to take place on a raw database export
#This does not need to be run again on the files included in the repo
import json

#Opens the json files with the built-in json module
#Uses the cp850 encoder to handle all of the rocket ship emojis
posts = json.load(open('./posts.json',encoding = 'cp850'))
comments = json.load(open('./comments.json',encoding='cp850'))

posts2 = {}
for index,post in enumerate(posts):
    posts2[index] = post
    
comments2 = {}
for index,comment in enumerate(comments):
    comments2[index] = comment
    
json.dump(posts2,open('./posts.json','w'))
json.dump(comments2,open('./comments.json','w'))
<!-- #endraw -->

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
#The [0:10] is subsetting the date string to only take YYYY-MM-DD
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
        #Handles the case where somebody put a $ in front of the stock ticker
        elif ('$'+ticker+' ' in text):
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
#Wrapping the previous date counter in a function for multiprocessing
#This counts up every stock's mention by the date it is mentioned on
def count_mentions_by_date(dates,posts,comments,stopwords,tickerToName,nameToTicker,mpqueue):
    #Counting up mentions of stocks by date
    stock_mentions = {date:{ticker.lower():0 for ticker in tickerToName.keys() if (ticker==ticker)} for date in dates}
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
    mpqueue.put(stock_mentions)
```

## Multiprocessing Note:

This _needs_ to be run on a Linux or Unix based-system (or maybe MacOS?) since multiprocessing works very differently in python on those platforms than it does on Windows. If you attempt to run the code below on Windows, it'll hang and never complete anything. On Linux it works fine.

<!-- #raw -->
from multiprocessing import cpu_count, Process, Queue

#Gathering the dates in the dataset
#The [0:10] is subsetting the date string to only take YYYY-MM-DD
dates = list(posts['date_posted'].map(lambda x: x[0:10]).unique())

#Counts up the number of cpu cores on the system
num_cpus = cpu_count()
#Holds the subsets of dates to be assigned to each cpu core
datelist = []
#Calculates how many dates to be handled by each cpu core, the final cpu will get any extras
dates_per_cpu = int(len(dates)/num_cpus)
#Getting the date split
for cpu in range(num_cpus):
    datelist.append(dates[dates_per_cpu*cpu:dates_per_cpu*(cpu+1)])
    

plist = []
qlist = []
#Setting up the necessary processes and starting their run
for cpu in range(num_cpus):
    qlist.append(Queue())
    plist.append(Process(target=count_mentions_by_date,
                         args=(datelist[cpu],posts,comments,stopwords,tickerToName,nameToTicker,qlist[cpu])))

    plist[cpu].start()
    
outlist = []
#Uses the queue's .get function which will pause execution of the main Python thread until it gets what it came for
#(the result from the Process)
for cpu in range(num_cpus):
    outlist.append(qlist[cpu].get())
                 
#Concatenates the result into one final dictionary mapping date to ticker to mention count
stock_mentions = {}
for out in outlist:
    stock_mentions = {**stock_mentions,**out}
<!-- #endraw -->

```python
#The above loop takes a really long time but that can be processed on a schedule later
#So, for testing purposes across different sessions, I dumped the results as a pickle
#to load in later
#pickle.dump(stock_mentions,open('./stock_mentions.p','wb'))
```

```python
stock_mentions = pickle.load(open('./stock_mentions.p','rb'))
```

```python
#total number of mentions regardless of date
total_mentions = {}
for date in stock_mentions:
    for ticker in stock_mentions[date]:
        if (ticker not in total_mentions):
            total_mentions[ticker] = stock_mentions[date][ticker]
        else:
            total_mentions[ticker] += stock_mentions[date][ticker]
            
#This sorts the total mentions            
#Only works properly in Python 3.7 or greater because dictionary keys are sorted in Python 3.7
total_mentions = {k: v for k, v in reversed(sorted(total_mentions.items(), key=lambda item: item[1]))}
```

```python
#Grabbing the first 12 tickers
top_tickers = list(total_mentions.keys())[0:12]

daily_counts = {}

for ticker in top_tickers:
    daily_counts[ticker] = []
    for date in dates:
        daily_counts[ticker].append(stock_mentions[date][ticker])
```

```python
import matplotlib.pyplot as plt

#Plotting one of the tickers
plt.plot(list(map(lambda x: x[5:],dates)),daily_counts['bb'])
```

# Time series forecasting

Below, a basic ARIMA forecast is done using the statsmodels library

```python
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

#Converts dates to datetime objects
dates_converted = list(map(lambda x: datetime.strptime(x,'%Y-%m-%d'),dates))

model = ARIMA(daily_counts['bb'],order=(8,3,4))

model_fit = model.fit()
```

```python
plt.plot(dates_converted,model_fit.predict(1,32),dates_converted,daily_counts['bb'])
```

```python
from datetime import timedelta

#a dictionary mapping each of the most mentioned stock tickers to the historical and predicted trends
ticker_graph_data = {}

#new_days is the number of days to predict forward
new_days = 7
#prediction dates stores the new dates that the model is predicting for
prediction_dates = [dates_converted[-1]+timedelta(1)]
#Adds on the new days using timedelta so that months/years are handled automatically
for day in range(new_days-1):
    prediction_dates.append(prediction_dates[-1] + timedelta(1))
    

for ticker in top_tickers:
    #Trains a new ARIMA model for every ticker
    model = ARIMA(daily_counts[ticker],order=(8,0,4))
    #Fits the model
    model_fit = model.fit()
    #Gets the predictions for the next 7 days after what is available in the daily_counts list
    predictions = model_fit.predict(len(daily_counts[ticker]),len(daily_counts[ticker])+new_days)
    
    #Below, the historical and forecasted data are put together into one dictionary structure, 
    #The dates are converted to "MM-DD"
    #The historical data
    #historical = [(str(dates_converted[i].month)+'-'+str(dates_converted[i].day),daily_counts[ticker][i]) for i in range(len(dates_converted))]
    historical = [(dates_converted[i].isoformat(),daily_counts[ticker][i]) for i in range(len(dates_converted))]
    #The predictions
    #prediction_data = [((str(prediction_dates[i].month)+'-'+str(prediction_dates[i].day)),predictions[i]) for i in range(len(prediction_dates))]
    prediction_data = [(prediction_dates[i].isoformat(),predictions[i]) for i in range(len(prediction_dates))]
    
    ticker_graph_data[ticker] = {'historical':historical,'prediction':prediction_data}
```

<!-- #raw -->
import json

json.dump(ticker_graph_data,open('./ticker_graph_data.json','w'))
<!-- #endraw -->

```python
#Plots out the final result calculated just as a sample
plt.figure(figsize=(20,20))
plt.plot(list(map(lambda x: x[0],historical)),list(map(lambda x: x[1],historical)),list(map(lambda x: x[0],prediction_data)),list(map(lambda x: x[1],prediction_data)))
```

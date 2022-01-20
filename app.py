import streamlit as st
import os
import pandas as pd
from io import BytesIO

from transformers import pipeline
import numpy as np
import pandas as pd
import tensorflow as tf
import ktrain
from ktrain import text

# SENTIMENT ANALYSIS 
from PIL import Image

image = Image.open('/images/pocket_aces_logo.jpg')
st.image(image)

st.title("SENTIMENT ANALYSIS")



  
  

import re
CLEANR = re.compile('<.*?>') 
CLEANR2 = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

def clean(raw_html):
  
  raw_html = raw_html.lower()
  cleantext = re.sub(CLEANR, '', raw_html)
  cleantext = re.sub(CLEANR2, '', cleantext)
  s = cleantext
  s = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", s)


  return s

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  df=pd.read_csv(uploaded_file)
  df1 = df

  st.write("Total No. of Comments: ",len(df))

  classifier = pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment")
  from transformers import AutoTokenizer, AutoModelForSequenceClassification
  
  if 'dframe' not in st.session_state:
    df["Pred"]=""
    #df["Cleaned_Comment"]=""
    for i in range(len(df)):
      print(i)
      sent = df['Comment'][i]
      sent=str(sent)
      #sent = re.sub("[^a-zA-Z]",  # Search for all non-letters
      #                      " ",          # Replace all non-letters with spaces
      #                      str(sent))
      sent = clean(sent)
      df["Comment"][i]=sent
      if len(sent)<160:
        d = classifier(sent)
        #d = cleanhtml(d)
        for result in d:
          #print(df['Comment'][i],end="    ")
          #print(sent,end="    ")
          if result['label']=="LABEL_0":
            #print("Negative")
            #label.append("Negative")
            df['Pred'][i] = "Negative"
          elif result['label']=="LABEL_1":
            #print("Neutral")
            #label.append("Neutral")
            df['Pred'][i] = "Neutral"
          if result['label']=="LABEL_2":
            #print("Positive")
            #label.append("Positive")
            df['Pred'][i] = "Positive"





    predictor = ktrain.load_predictor('/content/drive/MyDrive/Jan_11/prepr_Jan_12')

    for i in range(len(df)):
      print(i)
      if df["Pred"][i]!="Positive":
        data = clean(df["Comment"][i])
        df["Pred"][i] = predictor.predict(data)
      
    st.session_state['dframe'] = df

  #st.dataframe(df)

  #sentiment_analysis()

  # PIE CHART
  df = st.session_state.dframe
  p=0
  n=0
  ne=0
  for i in range(len(df)):
    if df["Pred"][i] == "Positive":
      p=p+1
    elif df["Pred"][i] == "Negative":
      n=n+1
    elif df["Pred"][i] == "Neutral":
      ne=ne+1

  import matplotlib.pyplot as plt
  import seaborn as sns
  import numpy as np

  st.header("SENTIMENT PIE-CHART")

  def func(pct, allvalues):
      absolute = int(pct / 100.*np.sum(allvalues))
      return "{:.1f}%\n".format(pct, absolute)

  #define data
  data = [p, n, ne]
  labels = ['Positive', 'Negative', 'Neutral']


  colors = ("blue","red","orange")
  fig = plt.figure(figsize =(10, 7))
  plt.pie(data, labels = labels,colors=colors,autopct = lambda pct: func(pct, data))
  
  # show plot
  st.pyplot(fig)

  # N-GRAMS

  import re
  import unicodedata
  import nltk
  from nltk.corpus import stopwords# add appropriate words that will be ignored in the analysis
  ADDITIONAL_STOPWORDS = ['covfefe']
  import matplotlib.pyplot as plt

  import nltk
  nltk.download('stopwords')
  nltk.download('wordnet')

  def basic_clean(text):
    """
    A simple function to clean up the data. All the words that
    are not designated as a stop word is then lemmatized after
    encoding and basic regex parsing are performed.
    """
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english') + ADDITIONAL_STOPWORDS
    text = (unicodedata.normalize('NFKD', text)
      .encode('ascii', 'ignore')
      .decode('utf-8', 'ignore')
      .lower())
    words = re.sub(r'[^\w\s]', '', text).split()
    return [wnl.lemmatize(word) for word in words if word not in stopwords]



  # Sentiment N-Grams

 

  sentiment = st.radio("Sentiment",('All','Positive', 'Neutral', 'Negative'))

  df_new = pd.DataFrame(index=range(30000))
  df_new["Comment"]=""
  #df['Filtered']=""
  j=0
  if sentiment=='All':
    df_new["Comment"]=df["Comment"]
  else:
    for i in range(len(df)):
      if df["Pred"][i]==sentiment:
        df_new["Comment"][j]=df["Comment"][i]
        j=j+1

  ########################################################################
  #st.dataframe(df_new,3000,1000)  
  import plotly.graph_objects as go

  fig = go.Figure(data=[go.Table(
      header=dict(values=list(df_new.columns),
                  fill_color='paleturquoise',
                  align='left'),
      cells=dict(values=df_new.transpose().values.tolist(),
                fill_color='lavender',
                align='left'))
  ])
  fig.update_layout(width=800, height=700)
  st.write(fig)
  #st.table(df_new,width=800,height=700)
  words = basic_clean(''.join(str(df_new['Comment'].tolist())))
  st.header("Frequently Occuring Words")

  num2 = st.selectbox(
      'Phrases Word Count - Sentiment',
      (1,2,3,4,5))
  num_1 = st.selectbox(
      'Total Occurences - Sentiment',
      (5,10,15,20,25,30,35,40))
  if(num2):
    grams = (pd.Series(nltk.ngrams(words, num2)).value_counts())[:num_1]
    gr1 = (pd.Series(nltk.ngrams(words, num2)))[:num_1]
    gr1 = gr1.tolist()
    grams = grams.tolist()
    for i in range(len(grams)):
      st.write(gr1[i]," ",grams[i])
    





  # BRAND MENTIONS

  st.header('BRAND/ACTOR ANALYSIS')

  brand = st.text_input('Brand Name', '')
  brand = brand.split(',')
  ctr=0
  for i in range(len(df)):
    df["Comment"][i] = clean(df["Comment"][i])
    sent = df["Comment"][i].split(" ")
    for word in brand:
      for word_sent in sent:
        if word.casefold()==word_sent.casefold():
          ctr=ctr+1
  st.write("No. of Brand Mentions: ",ctr)

  brand_sentiment = st.radio("Brand Sentiment",('All','Positive', 'Neutral', 'Negative'))

  #brand_sentiment = st.radio("Brand Sentiment",('All','Positive', 'Neutral', 'Negative'))

  if brand_sentiment=='All':
    df['Filtered']=""
    df["Filt_Label"]=""
    j=0
    for i in range(len(df)):
      #if df["Pred"][i]==brand_sentiment:
      sent = df['Comment'][i]
      sent = sent.split(' ')
      #print(sent)
      for word in brand:
        for word_sent in sent:
          if word.casefold()==word_sent.casefold():
            df["Filtered"][j] = df['Comment'][i]
            df["Filt_Label"][j] = df["Pred"][i]
            j=j+1
  else:
    df['Filtered']=""
    df["Filt_Label"]=""
    j=0
    for i in range(len(df)):
      if df["Pred"][i]==brand_sentiment:
        sent = df['Comment'][i]
        sent = sent.split(' ')
        #print(sent)
        for word in brand:
          for word_sent in sent:
            if word.casefold()==word_sent.casefold():
              df["Filtered"][j] = df['Comment'][i]
              df["Filt_Label"][j] = df["Pred"][i]
              j=j+1  


  words = basic_clean(''.join(str(df['Filtered'].tolist())))
  p=0
  n=0
  ne=0
  for j in range(len(df)):
    if df["Filt_Label"][j]=="Positive":
      p=p+1
    elif df["Filt_Label"][j]=="Negative":
      n=n+1
    elif df["Filt_Label"][j]=="Neutral":
      ne=ne+1
  
  def func(pct, allvalues):
      absolute = int(pct / 100.*np.sum(allvalues))
      return "{:.1f}%\n".format(pct, absolute)

  #define data
  data = [p, n, ne]
  labels = ['Positive', 'Negative', 'Neutral']


  colors = ("blue","red","orange")
  fig = plt.figure(figsize =(10, 7))
  plt.pie(data, labels = labels,colors=colors,autopct = lambda pct: func(pct, data))
  
  # show plot
  st.pyplot(fig)




  #num = st.number_input('N-Grams for Brands', 3)
  num = st.selectbox(
      'Phrases Word Count',
      (1,2,3,4,5))
  num_2 = st.selectbox(
      'Total Occurences',
      (5,10,15,20,25,30,35,40))
  
  if(num):
    grams = (pd.Series(nltk.ngrams(words, num)).value_counts())[:num_2]
    gr1 = (pd.Series(nltk.ngrams(words, num)))[:num_2]
    gr1 = gr1.tolist()
    grams = grams.tolist()
    for i in range(len(grams)):
      st.write(gr1[i]," ",grams[i])
    gr_names = pd.Series(nltk.ngrams(words, num))[:num_2]
    #gr = gr.tolist()
    gr_nums = (pd.Series(nltk.ngrams(words, num)).value_counts())[:num_2]
    gr_nums = gr_nums.tolist()
      #st.write(grams[i],end=" ")
    #st.bar_chart(gr1,grams)
    grnums=[]
    for i in gr_names:
      #print(type(i))
      i=str(i)
      #print(type(i))
      strings=""
      for j in i:
        strings+=j
        #print(j)
      
      grnums.append(strings)
      #st.bar_chart(grnums,gr_nums)
    dfr = pd.DataFrame(
        
        gr_nums,
        grnums,

    )
    #st.bar_chart(data = dfr)
  st.header("NOTABLE TIMESTAMPS")
  s=set([])
  l=[]
  for i in range(len(df1)):
  
    string=""
    flag=0
    sent = df1["Comment"][i]
    ctr=0
    for w in range(len(sent)-1):
      flag=0
      if sent[w]=='t' and sent[w+1]=='=':
        for j in range(w,len(sent)-1):
          if sent[j]=='>' and sent[j+1].isdigit()==True:
            flag=1
          if flag!=0:
            string=string+str(sent[j])
          if sent[j]=='<':
            flag=0
            ctr=ctr+1     
          
          
    if string!="":
      string=string[1:]
      string=string[:-1]
      string=string.split('<>')
      #print(string)
      for str_list in string:
        s.add(str_list)
        l.append(str_list)
  d={}
  for i in s:
    d[i]=l.count(i)
  for w in sorted(d, key=d.get, reverse=True):
      st.write(w, d[w]) 
  
  st.title("Sentiment with Time")
  from datetime import timedelta
  import datetime

  dstart = st.date_input("Enter the Start Date")
  
  dstart=datetime.datetime(dstart.year,dstart.month,dstart.day)

  dend = st.date_input("Enter the End Date")
  
  dend=datetime.datetime(dend.year,dend.month,dend.day)

  range_of_date = dend-dstart
  partition = range_of_date.days/4;
  day_add = partition
  pos=[0,0,0,0]
  neu=[0,0,0,0]
  neg=[0,0,0,0]
  prev_compare_date = dstart
  for j in range(4):
    compare_date = dstart+timedelta(days=day_add)
    for i in range(len(df)):
      d = df["Time"][i] 
      d=d[:10]
      day = int(d[8:10])
      month = int(d[5:7])
      year = int(d[:4])
      dt = datetime.datetime(year,month,day)
      if dt<compare_date and dt>=prev_compare_date:
        if df["Pred"][i]=="Positive":
          pos[j]=pos[j]+1
        elif df["Pred"][i]=="Neutral":
          neu[j]=neu[j]+1
        elif df["Pred"][i]=="Negative":
          neg[j]=neg[j]+1
    prev_compare_date = compare_date
    day_add=day_add+partition

  labels = ['G1', 'G2', 'G3','G4']
  
  w = 0.2
  x = np.arange(len(labels))  # the label locations
  x1 = [i+w for i in x]
  x2 = [i+w for i in x1]
    # the width of the bars

  fig, ax = plt.subplots()
  ax.bar(x  , pos, width=w, label='Positive', color=['blue', 'blue', 'blue', 'blue'])
  ax.bar(x1, neu, width=w, label='Neutral', color=['orange', 'orange', 'orange', 'orange'])
  ax.bar(x2, neg, width=w, label='Negative', color=['red', 'red', 'red', 'red'])

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('Scores')
  ax.set_title('Scores by group and gender')
  ax.set_xticks(x, labels)
  ax.legend()



  st.pyplot(fig)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yanickdupuisbinette
"""


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import re
import emoji

#Count vectorizer for N grams
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

# Nltk for tekenize and stopwords
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 


class TextHelper():

    def __init__(self):
        pass

    # check si il y a des missing value
    def missing_value_of_data(self, data):
        total=data.isnull().sum().sort_values(ascending=False)
        percentage=round(total/data.shape[0]*100,2)
        return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])


    def count_values_in_column(self, data,feature):
        total=data.loc[:,feature].value_counts(dropna=False)
        percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
        return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])


    def unique_values_in_column(self, data,feature):
        unique_val=pd.Series(data.loc[:,feature].unique())
        return pd.concat([unique_val],axis=1,keys=['Unique Values'])

    def duplicated_values_data(self, data):
        dup=[]
        columns=data.columns
        for i in data.columns:
            dup.append(sum(data[i].duplicated()))
        return pd.concat([pd.Series(columns),pd.Series(dup)],axis=1,keys=['Columns','Duplicate count'])

    def find_url(self, string):
        text = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',string)
        return "".join(text)

    def find_emoji(self, text):

        emo_text=emoji.demojize(text)
        line=re.findall(r'\:(.*?)\:',emo_text)
        return line


    def remove_emoji(self, text):
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)


    def find_email(self, text):
        line = re.findall(r'[\w\.-]+@[\w\.-]+',str(text))
        return ",".join(line)

    def find_hash(self, text):
        line=re.findall(r'(?<=#)\w+',text)
        return " ".join(line)

    def find_at(self, text):
        line=re.findall(r'(?<=@)\w+',text)
        return " ".join(line)

    def find_number(self, text):
        line=re.findall(r'[0-9]+',text)
        return " ".join(line)

    def find_phone_number(self, text):
        line=re.findall(r"\b\d{10}\b",text)
        return "".join(line)

    def find_year(self, text):
        line=re.findall(r"\b(19[40][0-9]|20[0-1][0-9]|2020)\b",text)
        return line

    def find_nonalp(self, text):
        line = re.findall("[^A-Za-z0-9 ]",text)
        return line

    def find_punct(self, text):
        line = re.findall(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]*', text)
        string="".join(line)
        return list(string)


    def stop_word_fn(self, text):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        non_stop_words = [w for w in word_tokens if not w in stop_words]
        stop_words= [w for w in word_tokens if w in stop_words]
        return stop_words

    def ngrams_top(self, corpus,ngram_range,n=None):
        """
        List the top n words in a vocabulary according to occurrence in a text corpus.
        """
        vec = CountVectorizer(stop_words = 'english',ngram_range=ngram_range).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        total_list=words_freq[:n]
        df=pd.DataFrame(total_list,columns=['text','count'])
        return df


    def rep(self, text):
        grp = text.group(0)
        if len(grp) > 1:
            return grp[0:1] # can change the value here on repetition

    def unique_char(self, rep,sentence):
        convert = re.sub(r'(\w)\1+', rep, sentence)
        return convert

    def find_dollar(self, text):
        line=re.findall(r'\$\d+(?:\.\d+)?',text)
        return " ".join(line)

    # Number greater than 930
    def num_great(self, text):
        line=re.findall(r'9[3-9][0-9]|[1-9]\d{3,}',text)
        return " ".join(line)

    # Number less than 930
    def num_less(self, text):
        only_num=[]
        for i in text.split():
            line=re.findall(r'^(9[0-2][0-0]|[1-8][0-9][0-9]|[1-9][0-9]|[0-9])$',i) # 5 500
            only_num.append(line)
            all_num=[",".join(x) for x in only_num if x != []]
        return " ".join(all_num)


    def or_cond(self, text,key1,key2):
        line=re.findall(r"{}|{}".format(key1,key2), text)
        return " ".join(line)

    def and_cond(self, text):
        line=re.findall(r'(?=.*do)(?=.*die).*', text)
        return " ".join(line)

    # mm-dd-yyyy format
    def find_dates(self, text):
        line=re.findall(r'\b(1[0-2]|0[1-9])/(3[01]|[12][0-9]|0[1-9])/([0-9]{4})\b',text)
        return line

    def only_words(self, text):
        line=re.findall(r'\b[^\d\W]+\b', text)
        return " ".join(line)

    def only_numbers(self, text):
        line=re.findall(r'\b\d+\b', text)
        return " ".join(line)

    def boundary(self, text):
        line=re.findall(r'\bneutral\b', text)
        return " ".join(line)

    def search_string(self, text,key):
        return bool(re.search(r''+key+'', text))

    def pick_only_key_sentence(self, text,keyword):
        line=re.findall(r'([^.]*'+keyword+'[^.]*)', text)
        return line

    def pick_unique_sentence(self, text):
        line=re.findall(r'(?sm)(^[^\r\n]+$)(?!.*^\1$)', text)
        return line

    def find_capital(self, text):
        line=re.findall(r'\b[A-Z]\w+', text)
        return line

    def add_all(self, df):
        df['url'] = df['text'].apply(lambda x: self.find_url(x))
        df['emoji'] = df['text'].apply(lambda x: self.find_emoji(x))
        df['text'] = df['text'].apply(lambda x: self.remove_emoji(x))
        df['email'] = df['text'].apply(lambda x: self.find_email(x))
        df['hash'] = df['text'].apply(lambda x: self.find_hash(x))
        df['at_mention'] = df['text'].apply(lambda x: self.find_at(x))
        df['number'] = df['text'].apply(lambda x: self.find_number(x))
        df['phone_number'] = df['text'].apply(lambda x: self.find_phone_number(x))
        df['year'] = df['text'].apply(lambda x: self.find_year(x))
        df['non_alp'] = df['text'].apply(lambda x: self.find_nonalp(x))
        df['punctuation'] = df['text'].apply(lambda x: self.find_punct(x))
        # df['stop_words'] = df['text'].apply(lambda x: self.stop_word_fn(x))

        df['dollar'] = df['text'].apply(lambda x: self.find_dollar(x))
        df['num_great'] = df['text'].apply(lambda x: self.num_great(x))
        df['num_less'] = df['text'].apply(lambda x: self.num_less(x))


        df['dates'] = df['text'].apply(lambda x: self.find_dates(x))
        df['only_words'] = df['text'].apply(lambda x: self.only_words(x))
        df['only_num'] = df['text'].apply(lambda x: self.only_numbers(x))

        df['pick_unique'] = df['text'].apply(lambda x: self.pick_unique_sentence(x))
        df['caps_word'] = df['text'].apply(lambda x: self.find_capital(x))
        df['text_length'] = df['text'].str.split().map(lambda x: len(x))
        df['char_length'] = df['text'].str.len()
        return df



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 19:09:18 2019

@author: mayankraj
"""

import numpy as np 
import pandas as pd 
pd.set_option('display.max_colwidth', -1)
from time import time
import re
import string
import os
import emoji
from pprint import pprint
import collections
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.3)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import gensim
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')
np.random.seed(37)

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.svm import SVC
#from sklearn.feature_extraction.text import TfidfVectorizer

class TextCounts(BaseEstimator, TransformerMixin):
    
    def count_regex(self, pattern, tweet):
        return len(re.findall(pattern, tweet))
    
    def fit(self, X, y=None, **fit_params):
        # fit method is used when specific operations need to be done on the train data, but not on the test data
        return self
    

    
    def transform(self, X, **transform_params):
        count_words = X.apply(lambda x: self.count_regex(r'\w+',str(x))) 
        count_mentions = X.apply(lambda x: self.count_regex(r'@\w+', str(x)))
        count_hashtags = X.apply(lambda x: self.count_regex(r'#\w+', str(x)))
        count_capital_words = X.apply(lambda x: self.count_regex(r'\b[A-Z]{2,}\b', str(x)))
        count_excl_quest_marks = X.apply(lambda x: self.count_regex(r'!|\?', str(x)))
        count_urls = X.apply(lambda x: self.count_regex(r'http.?://[^\s]+[\s]?',str( x)))
        # We will replace the emoji symbols with a description, which makes using a regex for counting easier
        # Moreover, it will result in having more words in the tweet
        count_emojis = X.apply(lambda x: emoji.demojize(str(x))).apply(lambda x: self.count_regex(r':[a-z_&]+:', str(x)))
        
        df = pd.DataFrame({'count_words': count_words
                           , 'count_mentions': count_mentions
                           , 'count_hashtags': count_hashtags
                           , 'count_capital_words': count_capital_words
                           , 'count_excl_quest_marks': count_excl_quest_marks
                           , 'count_urls': count_urls
                           , 'count_emojis': count_emojis
                          })
        
        return df
    


class CleanText(BaseEstimator, TransformerMixin):
    def removeMentions(self, inputText):
        return re.sub(r'@\w+', '', str(inputText))
    
    def removeUrls(self, inputText):
        #return re.sub(r'https?://[^\s] + [\s]?','', str(inputText))
        return re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', str(inputText))
    
    
    def emojiOneword(self, inputText):
        return inputText.replace('_','')
    
    def removePunctuations(self, inputText):
        #make translation table
        punct = string.punctuation
        trantab = str.maketrans(punct,len(punct)*' ')   # every punctuation is replaced bt space
        return inputText.translate(trantab)
    
    def removeDigits(elf, inputText):
        return re.sub('\d+','',str(inputText))
    
    def toLower(self, inputText):
        return inputText.lower()
    
    def removeStopwords(self, inputText):
        stopwordsList = stopwords.words('english')  #some words which might indicate a certain sentiment are retained via a whitelist
        whitelist = ["n't","not","no"]
        words = inputText.split()
        cleanWords = [word for word in words if(word not in stopwordsList or word in whitelist) and len(word) > 1]
        return " ".join(cleanWords)
    
    def stemming(self, inputText):
        porter = PorterStemmer()
        words = str(inputText).split()
        stemmedWords = [porter.stem(word) for word in words]
        return " ".join(stemmedWords)
    
    def fit(self, X, y = None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        clean_X = X.apply(self.removeDigits)
        clean_X = X.apply(self.removeMentions).apply(self.removeUrls).apply(self.emojiOneword).apply(self.removePunctuations).apply(self.removeDigits).apply(self.toLower).apply(self.removeStopwords).apply(self.stemming)
        #clean_X = X.apply(self.removeMentions)
        return clean_X
    

xls = pd.ExcelFile("trainingdata.xlsx", index = False);
df1 = pd.read_excel(xls,"Obama")
df2 = pd.read_excel(xls,"Romney")
df1 = df1.reset_index()
df1 = df1.drop(columns=['index'])
d1 =  df1.iloc[:,2:5]
d1 = d1.drop(d1.index[0])
df2 = df2.reset_index()
df2 = df2.drop(columns=['index'])
d2 = df2.iloc[:,2:5]
d2 = d2.drop(d2.index[0])
d1.columns = ['tweet','classes','your class']
d2.columns = ['tweet','classes','your class']
#sns.factorplot(x="classes", data=d1, kind="count", size=8, aspect=1.5, palette="PuBuGn_d")
#plt.show();
#d1 = d1.drop(d1[ (d1.classes == "!!!!") | (d1.classes == 2) | (d1.classes == "irrevelant")].index)
#d2 = d2.drop(d2[ (d2.classes == "!!!!") | (d2.classes == 2) | (d2.classes == "irrevelant")].index)
d1 = d1.loc[(d1.classes == 1) | (d1.classes == -1) | (d1.classes == 0)]
d2 = d2.loc[(d2.classes == 1) | (d2.classes == -1) | (d2.classes == 0)]

sns.factorplot(x="classes", data=d1, kind="count", size=8, aspect=0.5, palette="PuBuGn_d")
plt.show();
sns.factorplot(x="classes", data=d2, kind="count", size=8, aspect=0.5, palette="PuBuGn_d")
plt.show();

ct = CleanText()
ct1 = CleanText()
d1 = d1.astype({"classes": str})
d2 = d2.astype({"classes": str})
sr_clean = ct.fit_transform(d1.tweet)
sr2_clean = ct1.fit_transform(d2.tweet)
    #print(sr_clean.sample(5))
    
empty_clean = sr_clean == ''
print('{} records have no words left after text cleaning'.format(sr_clean[empty_clean].count()))
sr_clean.loc[empty_clean] = '[no_text]'

empty_clean = sr2_clean == ''
print('{} records have no words left after text cleaning'.format(sr2_clean[empty_clean].count()))
sr2_clean.loc[empty_clean] = '[no_text]'


tc = TextCounts()
df_eda = tc.fit_transform(d1.tweet)
df_eda['classes'] = d1.classes  

tc1 = TextCounts()
df2_eda = tc1.fit_transform(d2.tweet)
df2_eda['classes'] = d2.classes

cv = CountVectorizer()
bow = cv.fit_transform(sr_clean)
word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(20), columns = ['word', 'freq'])
fig, ax = plt.subplots(figsize=(18, 10))
sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
plt.show();
    
cv1 = CountVectorizer()
bow = cv1.fit_transform(sr2_clean)
word_freq = dict(zip(cv1.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(20), columns = ['word', 'freq'])
fig, ax = plt.subplots(figsize=(18, 10))
sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
plt.show();

df_model = df_eda
df_model['clean_text'] = sr_clean
df_model.columns.tolist()


df2_model = df2_eda
df2_model['clean_text'] = sr2_clean
df2_model.columns.tolist()


class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols
    def transform(self, X, **transform_params):
        return X[self.cols]
    def fit(self, X, y=None, **fit_params):
        return self
 
    
X_train, X_test, y_train, y_test = train_test_split(df_model.drop('classes', axis=1), df_model.classes, test_size=0.1, random_state=37)
y_train = y_train.astype('str')

#X_train, X_test, y_train, y_test = train_test_split(df2_model.drop('classes',axis = 1), df2_model.classes, test_size = 0.1, random_state = 37)
#y_train = y_train.astype('str')

def grid_vect(clf, parameters_clf, X_train, X_test, parameters_text=None, vect=None, is_w2v=False):
        
        textcountscols = ['count_capital_words','count_emojis','count_excl_quest_marks','count_hashtags'
                          ,'count_mentions','count_urls','count_words']
        
        if is_w2v:
            w2vcols = []
            for i in range(SIZE):
                w2vcols.append(i)
            features = FeatureUnion([('textcounts', ColumnExtractor(cols=textcountscols))
                                     , ('w2v', ColumnExtractor(cols=w2vcols))]
                                    , n_jobs=-1)
        else:
            features = FeatureUnion([('textcounts', ColumnExtractor(cols=textcountscols))
                                     , ('pipe', Pipeline([('cleantext', ColumnExtractor(cols='clean_text')), ('vect', vect)]))]
                                    , n_jobs=-1)
        
        pipeline = Pipeline([
            ('features', features)
            , ('clf', clf)
        ])
        
        # Join the parameters dictionaries together
        parameters = dict()
        if parameters_text:
            parameters.update(parameters_text)
        parameters.update(parameters_clf)
        # Make sure you have scikit-learn version 0.19 or higher to use multiple scoring metrics
        grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=10)
        
        print("Performing grid search...")
        print("pipeline:", [name for name, _ in pipeline.steps])
        print("parameters:")
        t0 = time()
        grid_search.fit(X_train, y_train)   #use proper train set for docs
        print("done in %0.3fs" % (time() - t0))
        print()
        print("Best CV score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
            
        print("Test score with best_estimator_: %0.3f" % grid_search.best_estimator_.score(X_test, y_test))     #put y2 values for romney tweets
        print("\n")
        print("Classification Report Test Data")
        print(classification_report(y_test, grid_search.best_estimator_.predict(X_test)))
                            
        return grid_search


parameters_vect = {
 'features__pipe__vect__max_df': (0.25, 0.5, 0.75),
 'features__pipe__vect__ngram_range': ((1, 1), (1, 2)),
 'features__pipe__vect__min_df': (1,2)
   }


    # Parameter grid settings for MultinomialNB
parameters_mnb = {
    'clf__alpha': (0.25, 0.5, 0.75)
   }
    
    
    # Parameter grid settings for LogisticRegression
parameters_logreg = {
    'clf__C': (0.25, 0.5, 1.0),
    'clf__penalty': ('l1', 'l2')
   }
#kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)




mnb = MultinomialNB()
svm1 = SVC(probability=True, kernel="linear", class_weight="balanced") 
logreg = LogisticRegression()
tfidfvect = TfidfVectorizer()
countvect = CountVectorizer()
#svm = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5, tol=None)



parameters_svm = {
        'svc__C':[0.01,0.1,1]
        }




best_svm_countvect = grid_vect(svm1, parameters_svm,X_train, X_test, parameters_text=parameters_vect, vect=countvect)


best_mnb_countvect = grid_vect(mnb, parameters_mnb, X_train, X_test, parameters_text=parameters_vect, vect=countvect)

best_logreg_countvect = grid_vect(logreg, parameters_logreg, X_train, X_test, parameters_text=parameters_vect, vect=countvect)

best_mnb_tfidf = grid_vect(mnb, parameters_mnb, X_train, X_test, parameters_text=parameters_vect, vect=tfidfvect)  

best_logreg_tfidf = grid_vect(logreg, parameters_logreg, X_train, X_test, parameters_text=parameters_vect, vect=tfidfvect)


SIZE = 25

X_train['clean_text_wordlist'] = X_train.clean_text.apply(lambda x : word_tokenize(x))
X_test['clean_text_wordlist'] = X_test.clean_text.apply(lambda x : word_tokenize(x))

model = gensim.models.Word2Vec(X_train.clean_text_wordlist
                 , min_count=1
                 , size=SIZE
                 , window=3
                 , workers=4)    

model.most_similar('plane', topn=3)

def compute_avg_w2v_vector(w2v_dict, tweet):
    list_of_word_vectors = [w2v_dict[w] for w in tweet if w in w2v_dict.vocab.keys()]
    
    if len(list_of_word_vectors) == 0:
        result = [0.0]*SIZE
    else:
        result = np.sum(list_of_word_vectors, axis=0) / len(list_of_word_vectors)
        
    return result


X_train_w2v = X_train['clean_text_wordlist'].apply(lambda x: compute_avg_w2v_vector(model.wv, str(x)))
X_test_w2v = X_test['clean_text_wordlist'].apply(lambda x: compute_avg_w2v_vector(model.wv, str(x)))

X_train_w2v = pd.DataFrame(X_train_w2v.values.tolist(), index= X_train.index)
X_test_w2v = pd.DataFrame(X_test_w2v.values.tolist(), index= X_test.index)

# Concatenate with the TextCounts variables
X_train_w2v = pd.concat([X_train_w2v, X_train.drop(['clean_text', 'clean_text_wordlist'], axis=1)], axis=1)
X_test_w2v = pd.concat([X_test_w2v, X_test.drop(['clean_text', 'clean_text_wordlist'], axis=1)], axis=1)

#best_logreg_w2v = grid_vect(logreg, parameters_logreg, X_train_w2v, X_test_w2v, is_w2v=True)

    
#  Project Title and Description : " Internet Movie Review Dataset "

### Descriptions of the project :
                   " The Internet Movie Reviews Dataset is a collection of movie reviews written by users, often sourced from platforms like IMDb. It’s designed to help researchers and developers to create and test models that can determine the emotional tone or sentiment of these reviews whether they are positive or a negative reviews. "
### Content: 
    " It contains text reviews from users about movies, typically collected from platforms like IMDb."
### Feature: 
      " The dataset usually contains a columns, for this dataset it can contains the feature that is reviews in text form, that represents the user's opinion about a movie."
### Target Variable: 
         " The dataset contain the target column , in this dataset the target variable should be Sentiment. This is often binary (positive/negative)."
### Size:
     "The dataset may consist of thousands of labeled reviews for training, testing, and validating the different models."

### Goals of the project :
              " To build models capable of understanding and classifying the sentiment expressed in movie reviews. you can explore different NLP techniques like breaking down text into smaller parts like (tokenization), using word representations (embeddings), and figuring out the overall mood of a sentence (sentiment scoring). It’s all about helping AI better understand human language and emotions. "


### CODE EXPLAINATION : 

## import the Necessary Libraries.

import pandas as pd  (To Read the file)
import numpy as np   (To used for the array)
from sklearn.preprocessing import LabelEncoder (To convert the categorical to numeric label in target variable)
from sklearn.model_selection import train_test_split  (To split the data into train and test data)


###  "Machine Learning Models : "

from sklearn.feature_extraction.text import TfidfVectorizer 

          (To used to convert text data into numerical features by calculating the TF-IDF (Term Frequency-Inverse Document Frequency) scores for each word, that can capturing their importance in the text.)

from sklearn.linear_model import LogisticRegression 

      ( is used for binary classification tasks based on input data.)

from sklearn.naive_bayes import MultinomialNB 

        ( is used to classify text or categorical data with word counts frequencies).

###  " Deep Learning Models : "

import tensorflow as tf 
      
      (tensorflow and keras: Used to build and train neural networks for tasks like image recognition, text analysis, or predictions.)

from keras.models import Sequential 
        
        (Sequential: Simplifies creating a linear stack {LIFO} of layers for neural networks.)
from keras.layers import Dense, Dropout 
            
            (Dense: A fully connected layer, used to learn patterns in data.  Dropout: Prevents overfitting by randomly disabling neurons during training.)
from transformers import BertTokenizer, TFBertForSequenceClassification 
            
            (transformers and TFBertForSequenceClassification: Pretrained transformer models (like BERT) are used for advanced NLP tasks like sentiment analysis or text classification. BertTokenizer: Prepares text for input into BERT models.)


###  "Evaluation : "

from sklearn.metrics import accuracy_score

    (accuracy_score: Measures how many predictions were correct out of the total predictions.)

from sklearn.metrics import precision_score

    (precision_score: Shows the percentage of correct positive predictions out of all predicted positives.)

from sklearn.metrics import recall_score

    (recall_score: Shows the percentage of correct positive predictions out of all actual positives.)

from sklearn.metrics import f1_score
    
    (f1_score: Combines precision and recall into a single score to balance both.)

from sklearn.metrics import classification_report

    (classification_report: Provides a summary of precision, recall, F1 score, and support for each class in a classification problem.)

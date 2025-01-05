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

### Upload the Reviews Dataset :

chunks = pd.read_csv('IMDB Dataset.csv', chunksize=1000, engine='python', on_bad_lines='skip')

    (it can read the reviews dataset , chunks can be used it can read 1000 rows at a time because data is too large it can easily run without any interrupted, and on_bad_lines can be used if in the dataset have an error so those line should be skip)
    
data = pd.concat(chunks, ignore_index=True)

     (we can concatenate those 1000 1000 rows of chunks are combine together )
data

### convert the string sentiment to binary value :

le = LabelEncoder()   

      (thus label encoder can be convert the categorical data like positive , negative into a binary or numerical value for the target variable )
data['sentiment'] = le.fit_transform(data['sentiment'])
data['sentiment']
data['sentiment'].value_counts()

     (it can count the total 0's and 1's in a given target col )

### Import the preprocessing libraries :

import nltk 

    (it can be for NLP tasks. like Tokenization , stemming, lemmatization, and analyzing linguistic structure. )
    
import re

      (used for working with regular expressions to search, match, and manipulate strings based on patterns.)
      
from nltk.tokenize import word_tokenize

    (used to split text into individual words or tokens.)
    
from nltk.corpus import stopwords

    (NLTK's stopwords are often removed from text like (is, am , are, have ...)
    
from nltk.stem import WordNetLemmatizer

    ( it can reduces the words to their base or dictionary form. useful for normalizing text and improving the quality of NLP models.)


### Download  Some Necessary Resources:

nltk.download('punkt')

    Downloads the Punkt tokenizer, used for tokenizing text into sentences or words.
    
nltk.download('stopwords')

    Downloads a list of common stopwords for text preprocessing.

nltk.download('wordnet')

    Downloads the WordNet lexical database, used for lemmatization and word relationships.


corpus = []
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

    initialize an empty list that is corpus , after that we can load all the common stopwords, then Creates a lemmatizer to reduce words to their base forms.

for i in range(len(data)):
    text = re.sub(r'[^a-zA-Z]', ' ', data['review'][i])                  [Removes non-alphabetic characters from the review text.]
    text = text.lower()                                                         [text can be convert into lower case ]
    tokens = word_tokenize(text)                                              [paragraph can tokenize into sentences]
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]      [Lemmatizes each word and removes stopwords.]
    review = ' '.join(tokens)                        [Joins the cleaned tokens back into a single string.]
    corpus.append(review)  

         after all of this we can initialize the feature and target col, then split the given data into train test split



### Apply TF-IDF 

    TF-IDF : to convert the text into vector form and to assign importance to words based on their frequency in a document.
    TF-IDF is better because it balances term importance and uniqueness effectively
    we can apply tf-idf and then train the model 



### Model's Train :
    After that we can apply the machine learning models (MultinomialNB & Logistic Regression models) to train & Predict the model

### Evaluation :

    after that we can evaluate the both model's performance by using accuracy, precision, recall, f1-score 


### Deep Learning  Model

from sklearn.decomposition import TruncatedSVD

      TruncatedSVD is used for dimensionality reduction in sparse data
from scipy.sparse import csr_matrix

      csr_matrix is used to create and handle sparse matrices efficiently, saving memory by storing only non-zero elements.

xtrain_vector_sparse = csr_matrix(xtrain_vector)
xtest_vector_sparse = csr_matrix(xtest_vector)


svd = TruncatedSVD(n_components= 500)

    initializes the dimensionality reduction model to reduce the feature space to 500 components.
xtrain_reduced = svd.fit_transform(xtrain_vector_sparse)
xtest_reduced = svd.transform(xtest_vector_sparse)


### dense model

    After that we Convert the text data into dense representation with a lower precision because in high precisiom it can gives an error (sparse representations where many values are zero. it's useful because they allow models to learn more nuanced patterns)

### Build a Simple Neural Network

neural_model = Sequential([
    Dense(128, activation='relu', input_shape=(xtrain_dense.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

      Sequential is used to build a linear stack of layers where each layer has exactly one input and one output.
      128 neurons in the first dense layer, using ReLU activation and taking input shape from xtrain_dense. A 50% dropout layer for regularization.
     64 neurons in the second dense layer with ReLU activation. Another 50% dropout layer for regularization.
     ReLU is used to introduce non-linearity and speed up learning by outputting positive values and zeroing out negatives.
     A final output layer with 1 neuron using a sigmoid activation for binary classification.


### Compile ;
neural_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
      Uses the Adam optimizer, which adjusts the learning rate during training for faster convergence. Binary crossentropy loss is used for binary classification tasks. Accuracy is tracked as a performance metric.

### Train
neural_model.fit(xtrain_dense, ytrain, epochs=10, batch_size=4, verbose=1)

    Trains the model on xtrain_dense (input) and ytrain (target) for 10 epochs with a batch size of 4. verbose=1 shows progress during training

### Test
neural_pred = (neural_model.predict(xtest_dense) > 0.5).astype(int)

    Predicts on xtest_dense, and the output is thresholded at 0.5 (for binary classification), converting probabilities to 0 or 1, and the results are cast to integers (astype(int)).


### Evaluate

    After that we can calculate the neural network model performance, by using accuracy, precision, recall, f1-score.

### Pre-trained BERT Model

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    Loads the pre-trained BERT tokenizer (bert-base-uncased), which converts input text into tokens that BERT can process.
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    Loads the pre-trained BERT model (bert-base-uncased) with a classification head, fine-tuned for sequence classification tasks (binary classification in this case, due to num_labels=2).

### Train Model

xtrain_bert = bert_tokenizer(xtrain, padding=True, truncation=True, max_length=128, return_tensors='tf')

    xtrain is tokenized using the BERT tokenizer with padding, truncation, and a maximum sequence length of 128. The tokenized output is returned as TensorFlow tensors (return_tensors='tf').

xtest_bert = bert_tokenizer(xtest, padding=True, truncation=True, max_length=128, return_tensors='tf')

    Similarly, xtest is tokenized with the same parameters and returned as TensorFlow tensors.

padding, truncated :

    Padding ensures consistent length, truncation cuts long sequences, and max-length sets the maximum token limit for input sequences.


### Compile the BERT model:

bert_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    Uses the Adam optimizer with a learning rate of 1*10^-5.  Uses sparse categorical crossentropy loss for multi-class classification. Tracks accuracy as the performance metric.

### Train the BERT Model:
bert_model.fit(xtrain_bert['input_ids'], np.array(ytrain), epochs=2, batch_size=2, verbose=1)

### Test the BERT Model:
bert_pred = np.argmax(bert_model.predict(xtest_bert['input_ids'])[0], axis=1)

    Uses np.argmax to convert the predicted probabilities into class labels, selecting the index with the highest probability from the model's output for xtest_bert['input_ids'].



### Evaluation :

    After that we can calculate the BERT Model Performance by using the accuracy, precision, recall, f1-score.


# Topic Modelling :

from gensim.corpora import Dictionary

    We use Gensim to analyze and extract useful information from large text data. 
tokenized_corpus = [review.split() for review in corpus]

    tokenized_corpus splits each review in corpus into a list of words.
dictionary = Dictionary(tokenized_corpus)

    builds a Gensim dictionary, where each unique word in the corpus is assigned a unique ID,


### create the Bag-of-Words (BoW) representation for each document

    We use Bag-of-Words (BoW) to represent text as a collection of word counts, simplifying the analysis by turning text into a numerical format that models word frequencies without considering word order.where the frequency of words is key.

corpus = [dictionary.doc2bow(review) for review in tokenized_corpus]

    converts each review into a list of tuples where each tuple contains a word's ID and its frequency in the review.
print("Sample Bag-of-Words for first review:", corpus[0])


###  LDA Topic Modelling :

from gensim.models import LdaModel

    Gensim's LDA model is used for discovering topics in large text corpora by analyzing word patterns and co-occurrences

lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=5, random_state=42)

    corpus=corpus: The Bag-of-Words corpus is passed as input. id2word=dictionary: The dictionary mapping word IDs to words is provided. num_topics=10: The model is set to find 10 topics. passes=5: The model will go through the entire corpus 5 times during training to improve accuracy. random_state=42: Ensures reproducibility by fixing the random seed.

topics = lda_model.print_topics(num_words=5)

    retrieves the top 5 words for each of the topics discovered by the LDA model.
for idx, topic in topics:
    print("Topic :", idx + 1)


### Visualization :

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

    pyLDAvis is a library used to visualize the topics generated by LDA models.

lda_vis = gensimvis.prepare(lda_model, corpus, dictionary)

     prepares the LDA model, corpus, and dictionary for visualization.
pyLDAvis.display(lda_vis)

### BERTopic for Modern Topic Modeling

    The BERTopic library is used for topic modeling using transformer-based embeddings like BERT. It creates interpretable topics by clustering the embeddings of text data, allowing for better understanding and visualization of topics compared to traditional methods like LDA.

bert_model = BERTopic(n_gram_range=(1, 2))

    This defines the range of n-grams (contiguous word sequences) to consider during topic modeling. It includes both unigrams (single words) and bigrams (two consecutive words), helping to capture more context in the topics.

### Evaluation :
we can evaluate the BERTOPIC MODEL by the graph plotting


dominant_topics = [max(lda_model.get_document_topics(corpus[i]), key=lambda x: x[1])[0] for i in range(len(corpus))]

    For each document in the corpus, it retrieves the most probable topic using lda_model.get_document_topics and selects the topic with the highest probability.

topic_genre_counts = data.groupby(['Topic', 'genre']).size().reset_index(name='Count')

    how many times each genre appears for each topic.


sns.scatterplot(data=topic_genre_counts, x='Topic', y='Count', hue='genre', palette='Set2', s=100)

    'genre' colors the points based on the genre, using the Set2 color palette.

plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')

    bbox_to_anchor=(1.05, 1) positions the legend outside the plot to the right. loc='upper left' places the legend at the upper-left corner of the bounding box.


from wordcloud import WordCloud

    where the size of each word represents its frequency or importance in a given text or dataset. It's commonly used for exploring and displaying the most common words in text data.


### Deliverables :

    After that we can visualize the rating for the positive and negative movie reviews. 

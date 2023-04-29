from pandas._libs.algos import pad
#data manipulation library
%matplotlib inline
!pip install -q keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import string
import re
import requests
import nltk
nltk.download('all')
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud,STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from bs4 import BeautifulSoup
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense,Input, Embedding,LSTM,Dropout,Conv1D, MaxPooling1D, GlobalMaxPooling1D,Dropout,Bidirectional,Flatten,BatchNormalization
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
import seaborn as sns



# data set importing
data = pd.read_csv('/content/data.csv', encoding = 'ISO-8859-1')
data['review'] = data['review'].astype(str)

data.head()
data.shape
data.info()
data.describe()
data['sentiment'].value_counts()
data.isnull().sum()
data.duplicated().sum()
data.drop_duplicates(inplace = True)
data.shape
stop = stopwords.words('english')
wl = WordNetLemmatizer()
mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot",
           "'cause": "because", "could've": "could have", "couldn't": "could not",
           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not",
           "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will",
           "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
           "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
           "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have",
           "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have"}
           #add more according to your requirement }








#function to clean data
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def preprocess(text):
    # Lowercase the text
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z]+', ' ', text)

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join the tokens back into a single string
    text = " ".join(tokens)

    return text

data['review']=data['review'].apply(preprocess)
data.head()

#converting target variable to numeric labels
data.sentiment = [ 1 if each == "positive" else 0 for each in data.sentiment]
data.head()
#splitting into train and test
train, test= train_test_split(data, test_size=0.2, random_state=111)
#train dataset
Xtrain, ytrain = train['review'], train['sentiment']
#test dataset
Xtest, ytest = test['review'].astype(str), test['sentiment']

print(Xtrain.shape,ytrain.shape)
print(Xtest.shape,ytest.shape)
#vectorizing the data for the conversion of text into number
vect = TfidfVectorizer()
print(vect)
Xtrain_vect= vect.fit_transform(Xtrain)
print(Xtrain_vect)
Xtest_vect = vect.transform(Xtest)
print(Xtest_vect)
count_vect = CountVectorizer() #Convert a collection of text documents to a matrix of token counts
Xtrain_count = count_vect.fit_transform(Xtrain)
Xtest_count = count_vect.transform(Xtest)

#lstm model
MAX_VOCAB_SIZE = 35174
tokenizer = Tokenizer(num_words = MAX_VOCAB_SIZE,oov_token="<oov>")
tokenizer.fit_on_texts(Xtrain)
word_index = tokenizer.word_index
#print(word_index)
V = len(word_index)
print("Vocabulary of the dataset is : ",V)

##create sequences of reviews
seq_train = tokenizer.texts_to_sequences(Xtrain)
seq_test =  tokenizer.texts_to_sequences(Xtest)

#choice of maximum length of sequences
seq_len_list = [len(i) for i in seq_train + seq_test]

#if we take the direct maximum then
max_len=max(seq_len_list)
print('Maximum length of sequence in the list: {}'.format(max_len))


# when setting the maximum length of sequence, variability around the average is used.
max_seq_len = np.mean(seq_len_list) + 2 * np.std(seq_len_list)
max_seq_len = int(max_seq_len)
print('Maximum length of the sequence when considering data only two standard deviations from average: {}'.format(max_seq_len))



#create padded sequences
pad_train=pad_sequences(seq_train,truncating = 'post', padding = 'pre',maxlen=max_seq_len)
print(pad_train)
pad_test=pad_sequences(seq_test,truncating = 'post', padding = 'pre',maxlen=max_seq_len)
print(pad_test)


#Splitting training set for validation purposes
Xtrain,Xval,ytrain,yval=train_test_split(pad_train,ytrain, test_size=0.2,random_state=10)

def Cnn_lstm_model(Xtrain,Xval,ytrain,yval,V,D,maxlen,epochs):
    print("----Building the model----")
    i = Input(shape=(maxlen,))
    x = Embedding(input_dim =  V + 1,output_dim = D,input_length = maxlen)(i) #Turns positive integers (indexes) into dense vectors of fixed size.
    # x = BatchNormalization(axis = 1)(x) #for the speed of training of neural network
    x = Dropout(rate= 0.2)(x) #one in five inputs will be randomly excluded from each update cycle.
    x = Conv1D( filters =100,kernel_size= 2,strides = 2, activation = 'relu',input_shape=(178, 1), padding='same')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling1D(pool_size =2,padding= 'same')(x) #to reduce the dimensions of the feature map maximum value for patches of a feature map
    #x = Bidirectional(LSTM(64,return_sequences=True))(x)
    x = LSTM(units=100,activation="tanh")(x) # lstm
    x = Dense(64, activation='sigmoid')(x) #dense layer will return an array of probability score
    x = Dropout(0.4)(x)
    #sigmoid --> For small values (<-5), sigmoid returns a value close to zero, and for large values (>5) the result of the function gets close to 1.
    x = Dense(1, activation='sigmoid')(x) # used to classify features of the input into various classes.
    model = Model(i, x)
    model.summary()
    #Training the LSTM
    print("----Training the network----")
    # binary_crossentropy for the multilabel classificatio   ,since our labels contains positive and negative
    model.compile(optimizer= Adam(learning_rate= 0.00001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
    filepath="weights_best_cnn.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
    callbacks_list = [checkpoint]

    r = model.fit(Xtrain,ytrain,
                  validation_data = (Xval,yval),
                  epochs = epochs,
                  verbose = 2,
                  batch_size = 32,
                  callbacks = callbacks_list)
    model.save('sentiment_analysis_model.h5')
    print("Train score:", model.evaluate(Xtrain,ytrain))
    print("Validation score:", model.evaluate(Xval,yval))
    n_epochs = len(r.history['loss'])

    return r,model,n_epochs
D = 80 #embedding dims
epochs = 3
r,model,n_epochs = Cnn_lstm_model(Xtrain,Xval,ytrain,yval,V,D,max_seq_len,epochs)

#saave the model
model.save("sentiment_analysis_model.h5")
def plotLearningCurve(history,epochs):
    epochRange = range(1,epochs+1)
    fig , ax = plt.subplots(nrows=1,ncols =2,figsize = (10,5))
    ax[0].plot(epochRange,history.history['accuracy'],label = 'Training Accuracy')
    ax[0].plot(epochRange,history.history['val_accuracy'],label = 'Validation Accuracy')
    ax[0].set_title('Training and Validation accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    ax[1].plot(epochRange,history.history['loss'],label = 'Training Loss')
    ax[1].plot(epochRange,history.history['val_loss'],label = 'Validation Loss')
    ax[1].set_title('Training and Validation loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    fig.tight_layout()
    plt.show()
plotLearningCurve(r,n_epochs)

print("Evaluate Model Performance on Test set")
result = model.evaluate(pad_test,ytest)
print(dict(zip(model.metrics_names, result)))

#Generate predictions for the test dataset
ypred = model.predict(pad_test)



#Get the confusion matrix
cf_matrix = confusion_matrix(ytest,ypred)
sns.heatmap(cf_matrix,cmap= "Blues",
            linecolor = 'black',
            linewidth = 1,
            annot = True,
            fmt='',
            xticklabels = ['Bad Reviews','Good Reviews'],
            yticklabels = ['Bad Reviews','Good Reviews'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

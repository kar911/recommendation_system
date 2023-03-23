import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pandas as pd
import re
# Download the stopwords corpus
nltk.download('stopwords')

# Download the punkt tokenizer
nltk.download('punkt')

# Define a sample text corpus
text = "This is an example sentence, showing off the stop words and punctuation removal."
def x(strr):
    # Remove punctuation
    text = strr.translate(str.maketrans('', '', string.punctuation))
    # Tokenize the text into individual words
    words = word_tokenize(text)
    # Remove stopwords

    stop_words = set(stopwords.words('english'))
    filtered_words = ' '.join([word for word in words if word.lower() not in stop_words])
    return re.sub(r'[^A-Za-z ]', '', filtered_words)


data=pd.read_csv("Entry Level Project Sheet - 3.1-data-sheet-udemy-courses-web-development.csv")

data['clean_sorted']= data["course_title"].apply(x)
data.to_csv("cleaned.csv")
# print(data)
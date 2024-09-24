

from processors.base_transformer import BaseTransformer

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
import string
import re
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from processors.base_transformer import BaseTransformer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from textblob import TextBlob
from transformers import pipeline
class SentimentAnalysisTransformer(BaseTransformer):
    
    def __init__(self):
        super().__init__()
        pass
    
    def transform(self, data, output_path):
        text = ''.join([x.get_text() for x in data.find_all('p')])
        text = text.lower()

        # Remove stop words

        stop = set(stopwords.words("english"))

        no_stop_words = [word for word in text.split() if word not in stop]
        no_brackets = [re.search(r'(.*)\[[0-9]*\]', word).group(1) if re.match(r"(.*)\[[0-9]*\].*", word) else word for word in no_stop_words]
        word_list = [word for word in no_brackets if not word.isdigit()]
        full_text = ' '.join(word_list)
        sent_tokens = sent_tokenize(full_text)
        print('polarity: ', [TextBlob(sentence).sentiment.polarity for sentence in sent_tokens])
        print('subjectivity: ', [TextBlob(sentence).sentiment.subjectivity for sentence in sent_tokens])
        sentiment_pipeline = pipeline("sentiment-analysis")
        print('transformer analysis: ', sentiment_pipeline(full_text))
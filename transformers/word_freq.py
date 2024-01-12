from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
import string
import re
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers.base_transformer import BaseTransformer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


class WordFreqTransformer(BaseTransformer):
    
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
        no_marks = [word for word in no_brackets if word not in set(string.punctuation)]
        translator = str.maketrans('', '', string.punctuation)
        no_marks = [word.translate(translator) for word in no_marks]
        word_list = [word for word in no_marks if not word.isdigit()]

        # stemming

        full_text = ' '.join(word_list)
        words = word_tokenize(full_text)
        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(word) for word in words]


        # counting word frequency

        wordfreq = []
        for w in lemmatized:
            wordfreq.append(lemmatized.count(w))

        def wordListToFreqDict(wordlist):
            wordfreq = [wordlist.count(p) for p in wordlist]
            return dict(list(zip(wordlist,wordfreq)))

        def sortFreqDict(freqdict):
            aux = [(freqdict[key], key) for key in freqdict]
            aux.sort()
            aux.reverse()
            return aux

        frq = wordListToFreqDict(lemmatized)
        sf = sortFreqDict(frq)

        words = []
        freq = []
        for s in sf:
            words.append(s[1])
            freq.append(s[0])

        df = pd.DataFrame({'word': words, 'freq': freq})

        df.to_csv(output_path, index=False)
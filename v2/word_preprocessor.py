import re
from nltk.tokenize import sent_tokenize
import nltk

nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
import simplemma


class WordPreprocessor:

    def __init__(self, sentences, language):
        self.sentences = sentences
        self.language = language
        self.stop_words = stopwords.words(language.lower())
        self.SIMPLEMMA_LANG_CODES = {
            "english": "en",
            "portuguese": "pt",
            "spanish": "es",
            "italian": "it",
            "german": "de",
            "french": "fr",
            "turkish": "tr",
            "dutch": "nl",
            "hungarian": "hu",
            "russian": "ru",
        }

    def to_lowercase(self):
        self.sentences = [x.lower() for x in self.sentences]

    def remove_digits(self):
        self.sentences = [re.sub(r"\d+", "", x) for x in self.sentences]

    def remove_punctuation(self):
        self.sentences = [re.sub(r"[^\w\s]", "", x) for x in self.sentences]

    def remove_stop_words(self):
        for sent in self.sentences:
            new_sent = []
            words = [x for x in sent.split()]
            for word in words:
                if word not in self.stop_words:
                    new_sent.append(word)
            self.sentences[self.sentences.index(sent)] = " ".join(new_sent)

    def lemmatize(self):
        new_setnencces = []
        for sent in self.sentences:
            words = [x for x in sent.split()]
            new_sent = []
            for word in words:
                new_sent.append(
                    simplemma.lemmatize(
                        word, lang=self.SIMPLEMMA_LANG_CODES[self.language.lower()]
                    )
                )
            new_sentences.append(" ".join(new_sent))
        self.sentences = new_sentences

    def preprocess(self):
        self.to_lowercase()
        self.remove_digits()
        self.remove_punctuation()
        self.remove_stop_words()
        self.lemmatize()
        return self.sentences

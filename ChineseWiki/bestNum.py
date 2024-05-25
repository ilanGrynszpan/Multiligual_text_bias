import os
import re
import jieba
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from typing import List, Tuple

# Remove Greek characters
def remove_greek_characters(text: str) -> str:
    greek_re = re.compile(r'[\u0370-\u03ff\u1f00-\u1fff]+')
    return greek_re.sub('', text)

# Load data
def load_data(filepath: str) -> List[str]:
    with open(filepath, 'r', encoding='utf-8') as file:
        data = [remove_greek_characters(line.strip()) for line in file if line.strip()]
    return data

# Segment data
def segment_data(data: List[str]) -> List[List[str]]:
    segmented_data = [list(jieba.cut(line)) for line in data]
    return segmented_data

# Create dictionary and corpus
def create_dictionary_corpus(data: List[List[str]]) -> Tuple[corpora.Dictionary, List[List[Tuple[int, int]]]]:
    dictionary = corpora.Dictionary(data)
    corpus = [dictionary.doc2bow(text) for text in data]
    return dictionary, corpus

# Perform and save LDA
def perform_and_save_lda(dictionary: corpora.Dictionary, corpus: List[List[Tuple[int, int]]], output_filepath: str, num_topics: int = 10, passes: int = 10, alpha: str = 'auto', eta = None, num_words: int = 20) -> models.LdaModel:
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes, alpha=alpha, eta=eta, random_state=42, per_word_topics=True)
    topics = lda_model.print_topics(num_words=num_words)
    save_topics(topics, output_filepath)
    return lda_model

def save_topics(topics: List[Tuple[int, str]], filename: str) -> None:
    with open(filename, 'w', encoding='utf-8') as file:
        for topic in topics:
            file.write(str(topic) + '\n')

# Model complexity and coherence
def evaluate_model(lda_model: models.LdaModel, corpus: List[List[Tuple[int, int]]], dictionary: corpora.Dictionary, texts: List[List[str]]) -> None:
    print('Perplexity:', lda_model.log_perplexity(corpus))  # A measure of how good the model is. Lower is better.
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Coherence Score:', coherence_lda)

class TopicModelling:
    @staticmethod
    def load_data(filepath: str) -> List[str]:
        return load_data(filepath)

    @staticmethod
    def segment_data(data: List[str]) -> List[List[str]]:
        return segment_data(data)

    @staticmethod
    def create_dictionary_corpus(data: List[List[str]]) -> Tuple[corpora.Dictionary, List[List[Tuple[int, int]]]]:
        return create_dictionary_corpus(data)

    @staticmethod
    def perform_and_save_lda(dictionary: corpora.Dictionary, corpus: List[List[Tuple[int, int]]], output_filepath: str, num_topics: int = 10, passes: int = 10, alpha: str = 'auto', eta = None, num_words: int = 20) -> models.LdaModel:
        return perform_and_save_lda(dictionary, corpus, output_filepath, num_topics, passes, alpha, eta, num_words)

    @staticmethod
    def evaluate_model(lda_model: models.LdaModel, corpus: List[List[Tuple[int, int]]], dictionary: corpora.Dictionary, texts: List[List[str]]) -> None:
        evaluate_model(lda_model, corpus, dictionary, texts)

class FindLDA:
    def __init__(self, dictionary: corpora.Dictionary, corpus: List[List[Tuple[int, int]]], texts: List[List[str]], output_dir: str):
        self.dictionary = dictionary
        self.corpus = corpus
        self.texts = texts
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def find_topic_numbers(self, start: int = 4, end: int = 12, passes: int = 10, alpha: str = 'auto', eta = None, num_words: int = 20):
        for num_topics in range(start, end + 1):
            output_filepath = os.path.join(self.output_dir, f'topicNum{num_topics}_words{num_words}.txt')
            lda_model = models.LdaModel(self.corpus, num_topics=num_topics, id2word=self.dictionary, passes=passes, alpha=alpha, eta=eta, random_state=42, per_word_topics=True)
            topics = lda_model.print_topics(num_words=num_words)
            save_topics(topics, output_filepath)

if __name__ == "__main__":
    file_path = '/Users/my/PycharmProjects/Multiligual-text-bias-/ChineseWiki/SegmentationResult/tokens.txt'
    raw_data = TopicModelling.load_data(file_path)
    segmented_data = TopicModelling.segment_data(raw_data)
    dictionary, corpus = TopicModelling.create_dictionary_corpus(segmented_data)
    test_output_dir = '/Users/my/PycharmProjects/Multiligual-text-bias-/ChineseWiki/TestTopicNum'

    # Experiment with different num_words values
    for num_words in [10, 20, 30]:
        find_lda = FindLDA(dictionary, corpus, segmented_data, test_output_dir)
        find_lda.find_topic_numbers(start=4, end=12, num_words=num_words)

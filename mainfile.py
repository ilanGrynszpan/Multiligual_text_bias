import requests
from bs4 import BeautifulSoup
from processors.links_collector import LinksCollector
from processors.sentiment_analysis import SentimentAnalysisTransformer
from processors.word_freq import WordFreqTransformer

searches = open("input/searches.csv", "r").read().split(',')

def run(searches):
    for search in searches:
        response = requests.get(
            url="https://en.wikipedia.org/wiki/"+search,
        )

        soup = BeautifulSoup(response.content, 'html.parser')

        output_paths = [
            "output/"+search+"_word_freq.csv",
            "output/"+search+"_links.csv"
        ]
        wf = WordFreqTransformer()
        lc = LinksCollector()
        sa = SentimentAnalysisTransformer()
        transformers = [sa]
        [transformer.transform(soup, output_path) for transformer, output_path in zip(transformers, output_paths)]

run(searches)

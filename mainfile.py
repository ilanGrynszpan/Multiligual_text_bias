import requests
from bs4 import BeautifulSoup
from transformers.links_collector import LinksCollector
from transformers.word_freq import WordFreqTransformer

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
        transformers = [wf, lc]
        [transformer.transform(soup, output_path) for transformer, output_path in zip(transformers, output_paths)]

run(searches)

response = requests.get(
    url="https://en.wikipedia.org/wiki/"+'democracy',
)

soup = BeautifulSoup(response.content, 'html.parser')

transformer = WordFreqTransformer()
transformer.transform(soup, "output/"+'democracy'+"_word_freq.csv")

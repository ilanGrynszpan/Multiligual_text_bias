from bs4 import BeautifulSoup
import pandas as pd
from transformers.base_transformer import BaseTransformer
class LinksCollector(BaseTransformer):
    
    def __init__(self):
        super().__init__()
        pass
    
    def transform(self, data, output_path):
        links = data.find_all('a')
        link_text = []
        for link in links:
            if link.has_attr('href'):
                if link['href'].startswith('/wiki/'):
                    link_text.append(link.get_text().lower())
        dict = {'links': link_text}
        df = pd.DataFrame(dict)
        print(df)
        df.to_csv(output_path, index=False)

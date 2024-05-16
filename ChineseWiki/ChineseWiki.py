import requests
from bs4 import BeautifulSoup
import re
import jieba


# get text
def get_text(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = [x.get_text() for x in soup.find_all('p')]
    cleaned_text = [re.sub(r'[\[\]0-9]*', '', x) for x in paragraphs]

    combined_text = ' '.join(cleaned_text)
    with open("saved_text.txt", "w", encoding="utf-8") as file:
        file.write(combined_text)
    return ' '.join(cleaned_text)

# sentence segmentation considering Chinese punctuation and ellipses
def split_sentences(text):
    sentences = re.split(r'(。|！|\!|\.|？|\?|……)', text)
    sentences = [sentence + delimiter for sentence, delimiter in zip(sentences[0::2], sentences[1::2])]
    return sentences

# remove punctuation and stopwords
def remove_punctuation_and_stopwords(sentences, stopwords_file='/stopwords_List/'):
    with open(stopwords_file, 'r', encoding='utf-8') as file:
        stopwords = set(file.read().splitlines())
    punctuation = "。！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
    sentences_clean = [''.join(jieba.cut(''.join([char for char in sent if char not in punctuation and char not in stopwords]))) for sent in sentences]
    with open('cleaned_sentences.txt', 'w', encoding='utf-8') as file:
        for sentence in sentences_clean:
            file.write(sentence + '\n')
    return sentences_clean

# tokenize sentences
def tokenize_sentences(sentences_clean):
    tokens = [word for sentence in sentences_clean for word in jieba.cut(sentence)]
    with open('tokens.txt', 'w', encoding='utf-8') as file:
        for token in tokens:
            file.write(token + '\n')
    return tokens


# save counts of sentences and words
def save_counts(sentences, tokens):
    with open('counts.txt', 'w', encoding='utf-8') as file:
        file.write(f'Sentence Count: {len(sentences)}\n')
        file.write(f'Word Count: {len(tokens)}\n')

# save unique words and its count
def save_unique_word_count(tokens):
    unique_words = set(tokens)
    with open('unique_token.txt', 'w', encoding='utf-8') as file:
        for token in tokens:
            file.write(token + '\n')
    unique_word_count = len(unique_words)
    print(f'Unique Word Count:{len(unique_words)}')
    with open('unique_word_counts.txt', 'w', encoding='utf-8') as file:
        file.write(f'Unique Word Count: {unique_word_count}\n')  # Writes the unique word count
    return unique_word_count



sample_page = "https://zh.wikipedia.org/wiki/民主"
web_text = get_text(sample_page)
sentences = split_sentences(web_text)
print(f'Sentence Count: {len(sentences)}')
sentences_clean = remove_punctuation_and_stopwords(sentences)
tokens = tokenize_sentences(sentences_clean)
print(f'Word Count: {len(tokens)}')
save_counts(sentences, tokens)
save_unique_word_count(tokens)

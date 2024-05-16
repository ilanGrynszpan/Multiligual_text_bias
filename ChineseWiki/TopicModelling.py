import re
import jieba
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# remove Greek characters
def remove_greek_characters(text):
    greek_re = re.compile(r'[\u0370-\u03ff\u1f00-\u1fff]+')
    return greek_re.sub('', text)

# Load data
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = [remove_greek_characters(line.strip()) for line in file if line.strip()]
    return data

# Segment data
def segment_data(data):
    segmented_data = [list(jieba.cut(line)) for line in data]
    return segmented_data

# Create dictionary and corpus
def create_dictionary_corpus(data):
    dictionary = corpora.Dictionary(data)
    corpus = [dictionary.doc2bow(text) for text in data]
    return dictionary, corpus

# Perform LDA
def perform_and_save_lda(dictionary, corpus, output_filepath, num_topics=10, passes=10, alpha='auto', eta=None, num_words=20):
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes, alpha=alpha, eta=eta, random_state=42, per_word_topics=True)
    topics = lda_model.print_topics(num_words=num_words)
    save_topics(topics, output_filepath)
    return lda_model


def save_topics(topics, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for topic in topics:
            file.write(str(topic) + '\n')

#model complexity and coherence
def evaluate_model(lda_model, corpus, dictionary, texts):
    print('Perplexity:', lda_model.log_perplexity(corpus))  # A measure of how good the model is. Lower is better.
    coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Coherence Score:', coherence_lda)

# Visualization
def visualize_topics(lda_model, num_topics, num_words=20):
    num_cols = min(num_topics, 5)
    fig, axes = plt.subplots(1, num_cols, figsize=(20, 4))

    # Chinese font for rendering
    font_path = 'ChineseFont.ttf'

    for topic_id, topic in enumerate(lda_model.print_topics(num_words=num_words)):
        topic_words = " ".join([word.split("*")[1].strip().strip('"') for word in topic[1].split(" + ")])
        wordcloud = WordCloud(width=800, height=800, font_path=font_path, random_state=21, max_font_size=110).generate(topic_words)
        col_id = topic_id % num_cols
        axes[col_id].imshow(wordcloud, interpolation="bilinear")
        axes[col_id].axis("off")
        axes[col_id].set_title("Topic: {}".format(topic_id))

    plt.tight_layout()
    plt.show()

# the optimal number of topics
def find_optimal_topics(dictionary, corpus, texts):
    coherence_values = []
    model_list = []
    for num_topics in range(2, 41, 2):
        model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, alpha='auto', random_state=42)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(round(coherencemodel.get_coherence(), 3))

    # coherence scores
    x = range(2, 41, 2)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.legend(("Coherence Values"), loc='best')
    plt.show()

    return model_list, coherence_values


if __name__ == "__main__":
    file_path = '/Users/my/PycharmProjects/ChineseWiki/SegmentationResult/tokens.txt'
    raw_data = load_data(file_path)
    segmented_data = segment_data(raw_data)
    dictionary, corpus = create_dictionary_corpus(segmented_data)
    output_filepath = '/Users/my/PycharmProjects/ChineseWiki/topics.txt'
    lda_model = perform_and_save_lda(dictionary, corpus, output_filepath, num_topics=10, num_words=20)  # Now showing 20 words per topic
    evaluate_model(lda_model, corpus, dictionary, segmented_data)
    visualize_topics(lda_model, num_topics=10)
    find_optimal_topics(dictionary, corpus, segmented_data)

from nltk.tokenize import sent_tokenize
import nltk

nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
import gensim
import gensim.corpora as corpora
from openai import OpenAI
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt


class TopicModeller:

    def __init__(self, sentences, language, num_topics):
        self.sentences = sentences
        self.language = language
        self.num_topics = num_topics
        self.LDA_model = None

    def model(self):

        def generate_ngrams(sentence):
            words = sentence.split()
            ngrams = []

            # Unigrams (individual words)
            ngrams.extend(words)

            # Bigrams (two consecutive words)
            bigrams = [" ".join(words[i : i + 2]) for i in range(len(words) - 1)]
            ngrams.extend(bigrams)

            # Trigrams (three consecutive words)
            trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
            ngrams.extend(trigrams)

            return ngrams

        docs = []
        for sent in self.sentences:
            docs.append(generate_ngrams(sent))
        print("------- docs --------")
        print(docs)
        print("---------------------")
        id2word = corpora.Dictionary(docs)
        print("------- id2word --------")
        print(id2word)
        print("---------------------")
        texts = docs
        corpus = [id2word.doc2bow(text) for text in texts]
        lda_model = gensim.models.ldamodel.LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=self.num_topics,
            random_state=100,
            update_every=1,
            chunksize=100,
            passes=10,
            alpha="auto",
            per_word_topics=True,
        )
        self.LDA_model = lda_model
        return lda_model

    def plotWordCloud(self, num_words=20, num_cols=5, savefig=False, figname=""):
        ## code from https://www.analyticsvidhya.com/blog/2023/02/topic-modeling-using-latent-dirichlet-allocation-lda/

        fig, axes = plt.subplots(
            int(self.num_topics / num_cols), num_cols, figsize=(20, 4)
        )

        for topic_id, topic in enumerate(
            self.LDA_model.print_topics(num_words=num_words)
        ):
            topic_words = " ".join(
                [word.split("*")[1].strip() for word in topic[1].split(" + ")]
            )
            wordcloud = WordCloud(
                width=800, height=800, random_state=21, max_font_size=110
            ).generate(topic_words)
            col_id = topic_id % num_cols
            axes[col_id].imshow(wordcloud, interpolation="bilinear")
            axes[col_id].axis("off")
            axes[col_id].set_title("Topic: {}".format(topic_id))

        plt.tight_layout()

        if not savefig:
            plt.show()
        else:
            plt.savefig(figname)

    def summarizeTopics(self, prompt, key, save=False, filename=""):
        client = OpenAI(api_key=key)
        prompt = "{0}: {1}".format(prompt, self.LDA_model.print_topics())
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[{"role": "system", "content": prompt}]
        )

        if save:
            with open(filename, "w") as f:
                f.write(response.choices[0].message.content)
        return response.choices[0].message.content

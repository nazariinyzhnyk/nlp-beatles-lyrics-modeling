import os
import argparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from preprocessing import get_dataframe, set_seed

pd.set_option('display.max_columns', 5)

nltk.download('stopwords')
nltk.download('wordnet')


class LemmaTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`', '(', ')']

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]


def main(query='all is love'):
    set_seed(42)

    df = get_dataframe(os.path.join('..', 'data', 'lyrics'))
    df.drop(df[(df.song == "All_You_Need_Is_Love.txt") & (df.album == "YellowSubmarine")].index, inplace=True)
    df.drop(df[(df.song == "Yellow_Submarine.txt") & (df.album == "YellowSubmarine")].index, inplace=True)

    stop_words = stopwords.words('english')
    for word in query.split():
        if word in stop_words:
            stop_words.remove(word)
    stop_words = set(stop_words)

    tokenizer = LemmaTokenizer()
    token_stop = tokenizer(' '.join(stop_words))

    documents = list(df.text)

    vectorizer = TfidfVectorizer(stop_words=token_stop,
                                 tokenizer=tokenizer)

    doc_vectors = vectorizer.fit_transform([query] + documents)

    cosine_similarities = cosine_similarity(doc_vectors[0:1], doc_vectors).flatten()
    document_scores = [item.item() for item in cosine_similarities[1:]]

    top_df = df.copy()
    top_df['similarity_score'] = document_scores
    top_df.sort_values('similarity_score', ascending=False).head(5)

    print(top_df.sort_values('similarity_score', ascending=False).head(5))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Document search ranking argparser.')
    parser.add_argument('--query', help='query to rank documents with', default='all is love')
    args = parser.parse_args()
    main(query=args.query)

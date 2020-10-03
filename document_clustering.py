import os
import argparse
import re

import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from preprocessing import get_dataframe, set_seed


def tokenize_and_stem(text):
    stemmer = SnowballStemmer("english")
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


def word_cloud(texts, title=''):
    text = ' '.join(texts)
    wordcloud = WordCloud().generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()


def assign_clusters(df, clusters):
    df_clust = df.copy()
    df_clust['cluster'] = clusters
    word_cloud(df_clust.text, title='WordCloud: All Songs')
    for cluster in df_clust.cluster.unique():
        songs = list(df_clust[df_clust.cluster == cluster].song)
        songs = [s.replace('.txt', '') for s in songs]
        print(f'Cluster {cluster} has {len(songs)} unique songs:\n{", ".join(songs)}\n')
        word_cloud(df_clust[df_clust.cluster == cluster]['text'], title=f'WordCloud: Cluster #{cluster}')

    return df_clust


def main(n_clusters: int = 3):
    set_seed(42)

    df = get_dataframe(os.path.join('..', 'data', 'lyrics'))
    df.drop(df[(df.song == "All_You_Need_Is_Love.txt") & (df.album == "YellowSubmarine")].index, inplace=True)
    df.drop(df[(df.song == "Yellow_Submarine.txt") & (df.album == "YellowSubmarine")].index, inplace=True)

    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for i in df.text:
        allwords_stemmed = tokenize_and_stem(i)
        totalvocab_stemmed.extend(allwords_stemmed)

        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)

    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)

    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                       min_df=0.2, stop_words='english',
                                       use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(df.text)

    terms = tfidf_vectorizer.get_feature_names()
    dist = 1 - cosine_similarity(tfidf_matrix)
    km = KMeans(n_clusters=n_clusters)
    km.fit(tfidf_matrix)

    clusters = km.labels_.tolist()
    df_clust = assign_clusters(df, clusters)

    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    cluster_words = []
    for i in range(n_clusters):
        curr_cluster = []
        print("Cluster %d words:" % i, end='')

        for ind in order_centroids[i, :10]:  # replace 6 with n words per cluster
            print(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0], end=',')
            curr_cluster.append(vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0])
        cluster_words.append(curr_cluster)
        print()

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
    xs, ys = pos[:, 0], pos[:, 1]
    cluster_names = {i: ' | '.join(cluster_words[i]) for i in range(len(cluster_words))}
    df_plot = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=df.song))

    groups = df_plot.groupby('label')

    fig, ax = plt.subplots(figsize=(17, 9))
    ax.margins(0.05)

    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                label=cluster_names[name],
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        ax.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')

    ax.legend(numpoints=1)  # show legend with only 1 point

    # add label in x,y position with the label as the film title
    for i in range(len(df_plot)):
        ax.text(list(df_plot.x)[i], list(df_plot.y)[i], list(df_plot.title)[i], size=7)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Document clustering argparser.')
    parser.add_argument('--n_clusters', help='number of clusters to split documents', default='3')
    args = parser.parse_args()
    main(n_clusters=int(args.n_clusters))

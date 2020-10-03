import os

import pandas as pd

from preprocessing import load_lyrics

DATA_DIR = os.path.join('..', 'data', 'lyrics')


def remove_line_breaks(row):
    return row.replace('\n', ' ')


def get_dataframe(path: str = DATA_DIR, remove_breaks: bool = True) -> pd.DataFrame:
    """
    Function performs os.walk through data folder, reads songs, saves them to lists along with album name.
        Then retrieved data forms a dataframe to work with.
    :param path: path to data
    :param remove_breaks: flag, whether to remove '\n' specials from text column
    :return: pd.DataFrame to work with. Columns: 'album', 'song', 'text'
    """

    if not os.path.isdir(path):
        load_lyrics(os.path.dirname(path))

    songs = []
    song_texts = []
    album_names = []
    for album in os.walk(path):
        if album[0] == path:
            continue
        album_name = os.path.basename(album[0])
        for song in album[2]:
            with open(os.path.join(album[0], song), 'r') as f:
                song_text = f.read()
            songs.append(song)
            song_texts.append(song_text)
            album_names.append(album_name)

    df = pd.DataFrame(list(zip(album_names, songs, song_texts)), columns=['album', 'song', 'text'])

    if remove_breaks:
        df.text = df.text.apply(remove_line_breaks)

    return df


if __name__ == '__main__':
    df = get_dataframe(DATA_DIR)
    print(df.head())

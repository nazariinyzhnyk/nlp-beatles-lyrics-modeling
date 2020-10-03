import os
import shutil
import warnings

import pygit2

DATA_DIR = os.path.join('..', 'data')


def load_files(src: str = "https://github.com/tylerlewiscook/beatles-lyrics.git", dest: str = DATA_DIR) -> None:
    """
    Loads files from src to dest within given git link
    :param src: git link to clone
    :param dest: destination path
    :return: None
    """
    pygit2.clone_repository(src, dest)


def clear_repo(path: str = DATA_DIR, dir_to_leave: str = 'lyrics') -> None:
    """
    Removes unnecessary files and directories (including .git dir) from working data folder
    :param path: path to delete files from
    :param dir_to_leave:
    :return: None
    """
    files = os.listdir(path)
    for f in files:
        if f != dir_to_leave:
            f = os.path.join(path, f)
            if os.path.isfile(f) or os.path.islink(f):
                os.remove(f)  # remove the file
            elif os.path.isdir(f):
                shutil.rmtree(f)  # remove dir and all files it contains
            else:
                warnings.warn(f'Could not delete file within path specified: {f}')


def load_lyrics(path=DATA_DIR):
    """
        Deletes DATA_DIR to ensure we get the right data.
        Then calls load_files() and clear_repo().
    """
    if os.path.isdir(path):
        shutil.rmtree(path)
    load_files(dest=path)
    clear_repo(path=path)


if __name__ == '__main__':
    load_lyrics()

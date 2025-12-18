"""
Author: Jose Varela
Email: jvarela@haverford.edu
This is my file containing helper functions for loading the dataset properly from a local directory

"""
import os
import librosa
import numpy as np
from scipy.io import wavfile

DATASET_DIR = os.path.expanduser("~/gtzan_project/data/genres_original")

GENRES = sorted([
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
])

GENRE_TO_INDEX = {genre: i for i, genre in enumerate(GENRES)}

def load_gtzan():
    """Load GTZAN manually from local directories."""
    data = []

    for genre in GENRES:
        genre_dir = os.path.join(DATASET_DIR, genre)

        for fname in os.listdir(genre_dir):
            if not fname.endswith(".wav"):
                continue

            file_path = os.path.join(genre_dir, fname)

            try:
                y, sr = librosa.load(file_path, sr=None)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

            label = GENRE_TO_INDEX[genre]
            data.append((y, label))

    return data

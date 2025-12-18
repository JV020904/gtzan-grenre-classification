"""
Author: Jose Varela
Email: jvarela@haverford
This file generates all features for the GTZAN dataset.
"""

"""
Author: Jose Varela
Email: jvarela@haverford
This file generates features for the GTZAN dataset.
"""

import os
import json
import numpy as np
from tqdm import tqdm

from utils import load_gtzan
from extract_features import feat_extract

FEATURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "features")
GTZAN_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'genres_original')


def main():
    print("Loading GTZAN dataset (local filesystem)...")
    data = load_gtzan()

    features = []
    labels = []

    print("Extracting features...")
    for y, label in tqdm(data):
        try:
            feat_vec = feat_extract(y)
        except Exception as e:
            print(f"Skipping file due to error: {e}")
            continue

        features.append(feat_vec)
        labels.append(label)

    # Convert lists to arrays
    X = np.array(features, dtype=np.float32)
    y = np.array(labels)

    # Build label → genre-name mapping
    genre_names = sorted(os.listdir(GTZAN_DIR))
    label_map = {i: name for i, name in enumerate(genre_names)}

    # Save to disk
    os.makedirs(FEATURE_DIR, exist_ok=True)

    np.save(os.path.join(FEATURE_DIR, "X.npy"), X)
    np.save(os.path.join(FEATURE_DIR, "y.npy"), y)

    with open(os.path.join(FEATURE_DIR, "labels.json"), "w") as f:
        json.dump(label_map, f, indent=4)

    print(f"Saved extracted features to {FEATURE_DIR}/")
    print(f"Total usable files: {len(X)}")
    print("Saved labels.json with genre mappings.")


if __name__ == "__main__":
    main()

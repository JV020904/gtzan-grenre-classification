"""
Author: Jose Varela
This is my file that defines the constants used
"""

import os

# Existing configs (keep yours)
Sample_Rate = 22050
Duration = 30
N_MFCC = 20

# === NEW PATHS FOR PROJECT STRUCTURE ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Make sure these folders exist
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

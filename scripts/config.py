"""
Author: Jose Varela
This is my file that defines the constants used in this project.

"""

import os

# Existing configs 
#Sample Rate
Sample_Rate = 22050

#Duration (30 second clips)
Duration = 30
N_MFCC = 20

#New paths for the project structure(previous ones made some issues arrise)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

#Making sure these folders exist
os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

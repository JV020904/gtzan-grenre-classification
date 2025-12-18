"""
Author: Jose Varela
Email: jvarela@haverford.edu
"""
import sys
import os

#Using this to try to fix an error I was having
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_extract(): 
    print("Running feature extraction...")
    os.system("python3 scripts/generate_features.py")

def run_train():
    print("Running model training...")
    os.system("python3 scripts/train_model.py")

def run_evaluate():
    print("Running model evaluation...")
    os.system("python3 scripts/evaluate.py")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py extract   # extract MFCC/chroma/spectral features")
        print("  python main.py train     # train SVM / RF / Logistic Regression")
        print("  python main.py evaluate  # evaluate trained models")
        print("  python main.py all       # run entire pipeline")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "extract":
        run_extract()

    elif command == "train":
        run_train()

    elif command == "evaluate":
        run_evaluate()

    elif command == "all":
        run_extract()
        run_train()
        run_evaluate()

    else:
        print(f"Unknown command: {command}")
        print("Valid options: extract, train, evaluate, all")
        sys.exit(1)


if __name__ == "__main__":
    main()

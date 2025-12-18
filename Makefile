#Author: Jose Varela
#Email: jvarela@haverford.edu
#This is my Makefile that allows for me to extract features,train the model, 
#evaluate the results, and clean the generated files (if needed)

PYTHON = python3

#Default target that runs the full pipeline
all: extract train evaluate


#Extract the audio features from the GTZAN dataset using main.py
extract:
	$(PYTHON) main.py extract

#training the model(s)
train:
	$(PYTHON) main.py train

#Evaluate the trained models.
evaluate:
	$(PYTHON) main.py evaluate

#Run the entire piepline in a single command
run: all

#Remove the generated files(in case something does not generate properly or needs tweaking). 
clean:
	rm -rf features/*
	rm -rf models/*
	rm -f results/*.png
	rm -f results/*.txt
	rm -f results/*.json

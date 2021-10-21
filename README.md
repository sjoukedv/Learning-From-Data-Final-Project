# Learning-From-Data-Final-Project
This repository contains the final project which analyses data that represents over 35,000 articles about climate change that have been published in the time period around the first 25 Conference of the
Parties meetings (COP1 until 24).

## Install instructions
1. Install all dependencies listed in requirements.txt by running `pip install -r requirements.txt`.
2. Download additional static content `python -m spacy download en_core_web_sm` (SpaCy package) and `wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz && gzip -d cc.en.300.bin.gz`.
3. Make sure all data to be trained on is present in the `data` folder.
4. Run each model separately e.g. `python NaiveBayes.py [--test_file <COP24.filt3.sub.json>]`.

## Models
All models extend the `BaseModel` in (`BaseModel.py`). By default the COP editions are being read by the helper functions in `dataParser.py`. All files in the `data` folder are used for training. Additionally, a separate test file can be specified by passing the `--test_file <file>` argument.
### Naive Bayes
A baseline classic model using bag-of-words

### Support Vector Machine 
A classic model with optimized feature set

### LSTM
An optimized LSTM model with pretrained static embeddings 

### BERT
A fine-tuned pretrained language model 

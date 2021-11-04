# Learning-From-Data-Final-Project
This repository contains the final project which analyses data that represents over 35,000 articles about climate change that have been published in the time period around the first 25 Conference of the
Parties meetings (COP1 until 24).

## Install instructions
1. Install all dependencies listed in requirements.txt by running `pip install -r requirements.txt`.
2. Download additional static content `python -m spacy download en_core_web_sm` (SpaCy package) and `glove.6B.zip` from the [Glove Embeddings Repository](https://github.com/stanfordnlp/GloVe) and unzip the `glove.6B.300d.txt` to the root directory
3. Make sure all data to be trained on is present in the `data` folder.
4. Run a model e.g. `python NaiveBayes.py [options]`, e.g.

Option | Description
--- | ---
`-test, --test`  | Run predictions on test set (otherwise uses dev set)
`-load, --load_model` | Load existing model or perform training
`-cop COP, --cop COP` | Path to single COP edition to test (e.g. data/COP25.filt3.sub.json)
`-undersample, --undersample` | Value which indicates whether to downsample the data
`-model_number MODEL_NUMBER, --model_number MODEL_NUMBER` | Name of model which should be loaded

Pass the `-h` or `--help` parameter to view the full list of options.

## Models
All models extend the `BaseModel` in (`BaseModel.py`). By default the COP editions are being read by the helper functions in `dataParser.py`. All files in the `data` folder are used for training. Additionally, a separate test file (of a single COP edition) can be specified by passing the `-cop <file>, --cop <file>` argument.
### Naive Bayes
A baseline classic model using bag-of-words

### Support Vector Machine 
A classic model with optimized feature set

### LSTM
An optimized LSTM model with pretrained static embeddings 

### BERT
A fine-tuned pretrained language model 

### FastText
[Open-source, free, lightweight library](https://fasttext.cc/) that allows users to learn text representations and text classifiers. It works on standard, generic hardware. Models can later be reduced in size to even fit on mobile devices.

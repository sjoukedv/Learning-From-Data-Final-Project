# Political Orientation Prediction from Newspaper Headlines
This repository contains the learning from final project which analyses data that represents over 35,000 articles about climate change that have been published in the time period around the first 25 Conference of the Parties meetings (COP1 until 24).

## Install instructions
1. Install all dependencies listed in requirements.txt by running `pip install -r requirements.txt`.
2. Download additional static content `python -m spacy download en_core_web_sm` (SpaCy package) and `glove.6B.zip` from the [Glove Embeddings Repository](https://github.com/stanfordnlp/GloVe) and unzip `glove.6B.300d.txt` to the <b>root</b> directory
3. Make sure all data to be trained on is present in the `data` folder.
4. Run a model e.g. `python NaiveBayes.py [options]`, for example:

Option | Description
--- | ---
`-test, --test`  | Run predictions on test set (otherwise uses dev set)
`-load, --load_model` | Load existing model or perform training (e.g. -load 00)
`-cop COP, --cop COP` | Path to single COP edition to test (e.g. data/COP25.filt3.sub.json)
`-undersample, --undersample` | Value which indicates whether to downsample the data
`-model_number MODEL_NUMBER, --model_number MODEL_NUMBER` | Name of model which should be loaded

Pass the `-h` or `--help` parameter to view the full list of options.

## Repository structure
- `figures`, contains all the figures used in the report.
- `models`, contains saved models (if possible due to size limits) of trained models
- `results`, contains results of all experiments

## Models
All models extend the `BaseModel` in (`BaseModel.py`). By default the COP editions are being read by the helper functions in `dataParser.py`. All files in the `data` folder are used for training. Additionally, a separate test file (of a single COP edition) can be specified by passing the `-cop <file>, --cop <file>` argument.

- Naive Bayes, a baseline classic model using bag-of-words
- Support Vector Machine, a classic model with optimized feature set
- LSTM, an optimized LSTM model with pretrained static embeddings 
- BERT, a fine-tuned pretrained language model 
- FastText, [Open-source, free, lightweight library](https://fasttext.cc/) that allows users to learn text representations and text classifiers.

The models can be found by accessing the following link: https://drive.google.com/drive/folders/1JfQFrZX9uBOetMH5qlbwjZHWD_vZPEo5?usp=sharing
- Please download the model and put them in the corresponding folder under models/
- Unfortunately, due to a bug in keras, it is not possible to load trained Bert models. 

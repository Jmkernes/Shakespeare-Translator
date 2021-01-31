# Shakespeare-Translator

![alt text](https://github.com/jmkernes/Shakespeare-Translator/blob/main/where_art_thou.png?raw=true)

*Attention weights of block 2 of the final layer of the Decoder. Maybe "wherefore" would have been a better translation, but alas...*

-----------------------------------------------

As the title suggests, the goal of this project is to translate a sentence from english to shakespearian english. More specifically, we provide two types of inference

1) Faithful translation. This utilizes beam search to try to construct the most probable shakespearian translation given the input

2) Generative translation. This uses a temperature parameter to sample possible translations. It may not be as accurate, but certainly more fun to give the model some freedom. We also implement a basic Minimum Bayes Risk inference on these translations to try to pick the "best" translation.

The stucture of this repository is as follows:

* **Web_scraping/**
    There is no English->Shakespeare dataset, so I had to create one. That's the goal of this folder
    * README - this has more details than I'll mention here.
    * web_scraper - the guts of the folder. A scraper used to create our dataset
    * log.txt - the results and logs of the scrape.
* **data_clean/**
    The scraped data is separated by play number and needs to be cleaned... so here we give the ready-to-go .txt files
    * original.txt - shakespearian version
    * modern.txt - the normal english version
* **Tokenizers/**
    This folder can be generated in a few minutes by running the create_tokenizer scripts, but we include it for convenience. We only use the following 2048 token tokenizers.
    * modern2k.model
    * original2k.model
* **trained_models/**
    * tensorboard_logs - run tensorbord --logdir tensorboard_logs to see our training metrics, loss, curves, learning rate, etc.
    * translate_weights... - these two files hold the actual trained model
    * model_config.json - a dictionary containing the model configuration so we can easily reconstruct its architecture and load in the weights
* **pretraining/**
    * This folder can be skipped. If interested, read the contents for more details and code. This folder implements a semi-supervised pretraining task to try to learn the encoder weights. 
    We use the masked langauge model BERT task as our training objective (bi directional cloze task), whereby the model is fed an input sequence with randomly masked elements, and it has
    to predict the true values of those elements. In our experiments, we did not find that using pretrained embeddings improved performance.
* create_tokenizer.py - a script to train the SentencePiece tokenizer. Use it if you want to experiment with the vocab size. Note the shakespearian corpus is upper bounded to ~13k tokens.
* tests.py - straightforward description
* data_utils.py - takes care of loading and processing the data for training. Called by the train script.
* utils.py - some one-off useful functions, like a print bar, masking, and model prediction.
* train.py - script for training the model. Default parameters are in the base_model.sh bash script
* colab_trainer.ipynb - A jupyter notebook that is ready for use on google colab. A much easier way to train: you can use a GPU, view the tensorboard logs in real time, and easily tweak the model as it trains.
* inference.ipynb - A jupyter notebook for playing around with the trained model. Also contains the setup for serializing the model to a SavedModel format for serving (we don't use it since I don't have server to host from)
* translator.py - This is the fun one! I recommend running it. It's a simple python script where you can type in text and get back translations.

## translator.py

This is the main script, try running it. Below is a screenshot of what it looks like in the terminal

![alt text](https://github.com/jmkernes/Shakespeare-Translator/blob/main/prog_screenshot.png?raw=true)

One of the fun things you can do is use this as an insult generator. The Minimum Bayes Risk generator will generate 16 random translations, and order them based on similarity. The similarity score is computed as follows: Each sequence is compared to every other sequence using python's difflib module method SequenceMatcher, resulting in a similarity score. A given entry is scored based on how similar it is to every other entry. We then rank sequences in order from most similar (meaning they sequence resembles the "mean" sequence) to least similar. An alternative similarity test using a count vectorizer is at the end of the inference notebook. It doesn't work as well; it loses temporal alignment and favors repetitive nonsensical translations.

## Using the data

Please use it! I ran out of GPU power, so I couldn't fine tune more. I strongly recommend using the data however to try to do better! Any comments, questions, concerns please email me or open a pull request. Thanks!


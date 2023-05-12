# ChatBot-Pytorch
ChatBot using Python

## Description:
This is a ChatBot using Python. Using some basic Natural Language Processing (NLP) techniques.
```
>>> Let's chat! (type 'quit' to exit)
>>> You: Hey
>>> Segredo: Hello, thanks for visiting
>>> You: It's time to me to go.
>>> Segredo: OK, it's was a pleasure to meet you, bye!
```

## Learning Process:
* NLP Basics: Tokenization, Stemming, Bag Of Words
* How to preprocess the data with ```nltk``` to feed it to your neural net
* How to implement the feed-forward neural net in Pytorch and train it
* The implementation should be easy to follow for beginners and provide a basic understanding of chatbots
* The implementation is straightforward with a Feed Forward Neural net with 2 hidden layers.
* Customization for your own use case is super easy. Just modify ```intents.json``` with possible pattern and responses and re-run the training.

## Starting out:
First and foremost, you need to have Python installed and pip. After that you need to install PyTorch and dependencies.

```pip install pytorch```

After installing pytorch you need to install nltk and download it.

```pip install nltk```

To download it after nltk you need to open python command line:

```
$ python
>>> import nltk
>>> nltk.download('punkt')
```

After this your project will work correctly.

## Usage:
How this software works ? It's quite simple, you change the ```intents.json``` file the way you want (don't change the layout), and after that you train your BOT.

```$ python train.py```

After training your bot, you can start the chat, have some fun with it.

```$ python chat.py```

As I mentioned before, you can customize it for your own need. Just modify the ```intents.json``` with possible patterns and responses and re-run the training.

## Learn More:
In **Natural Language Processing** (NLP), a **bag of words** (BoW) is a method for representing text data as a collection of word counts, disregarding the order and structure of the words in the text. The idea behind BoW is to treat a document as a "bag" of its constituent words, ignoring grammar and word order but keeping track of the frequency of each word.

**Tokenization** is a fundamental technique in **Natural Language Processing** (NLP) that involves breaking down a stream of text into smaller, meaningful units called tokens. These tokens can be words, phrases, or even individual characters, depending on the level of granularity needed for the analysis.

**Stemming** is a process in **Natural Language Processing** (NLP) that involves reducing a word to its base or root form, called a stem, by removing its affixes (suffixes or prefixes). The goal of stemming is to reduce different forms of a word to a common base form, so that they can be treated as the same word during analysis.

## Authors:
``` Octávio Marques (Segredo)```

## License:
[MIT](https://choosealicense.com/licenses/mit/)
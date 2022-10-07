# Text Generation Project: Writing TV Scripts

This repository contains a text generator which works with a Recursive Neural Network (RNN) based on LSTM units. The [Seinfeld Chronicles Dataset from Kaggle](https://www.kaggle.com/datasets/thec03u5/seinfeld-chronicles) is used, which contains the complete scripts from the [Seinfield TV Show](https://en.wikipedia.org/wiki/Seinfeld).

The project is a modification of the [Character-level RNN](https://github.com/karpathy/char-rnn) [implemented by Andrej Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). I have used materials from the [Udacity Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891), which can be obtained in their original form in [project-tv-script-generation](https://github.com/mxagar/deep-learning-v2-pytorch/tree/master/project-tv-script-generation).

Even though the text that the trained model is able to generate doesn't make much sense, it seems it follows the general structure that the scripts from the dataset have:

```
Example, TBD.
```

... and that's with very few hours of tweaking and training, so I I'd say it's a good starting point.

Table of Contents:

- [Text Generation Project: Writing TV Scripts](#text-generation-project-writing-tv-scripts)
  - [How to Use This](#how-to-use-this)
    - [Overview of Files and Contents](#overview-of-files-and-contents)
    - [Dependencies](#dependencies)
  - [Some Brief Notes on RNNs and Their Application to Language Modeling](#some-brief-notes-on-rnns-and-their-application-to-language-modeling)
  - [Notes on the Text Generation Application](#notes-on-the-text-generation-application)
  - [Improvements, Next Steps](#improvements-next-steps)
  - [Interesting Links](#interesting-links)
  - [Authorship](#authorship)

## How to Use This

In order to use model, you need to install the [dependencies](#dependencies) and execute the notebook [dlnd_tv_script_generation.ipynb](dlnd_tv_script_generation.ipynb), which is the main application file that defines and trains the network.

Next, I give a more detailed description on the contents and the usage.

### Overview of Files and Contents

Altogether, the project directory contains the following files and folders:

```
.
├── Instructions.md                     # Original Udacity instructions
├── README.md                           # This file
├── data/                               # Dataset folder
│   └── Seinfeld_Scripts.txt            # Dataset: scripts
├── dlnd_tv_script_generation.ipynb     # Project notebook
├── helper.py                           # Utilities: load/preprocess data, etc.
├── models/                             # Folder where trained models are saved
├── problem_unittests.py                # Unit tests
└── requirements.txt                    # Dependencies
```

As already introduced, the notebook [dlnd_tv_script_generation.ipynb](dlnd_tv_script_generation.ipynb) takes care of almost everything. That file uses the following two scripts:

- [helper.py](helper.py), which contains utility functions related to data preprocessing and model persisting,
- and [problem_unittests.py](problem_unittests.py), which contains the definitions of the unit tests run across the whole notebook.

All in all, the following sections/tasks are implemented in the project notebook:

- The dataset is loaded and briefly explored.
- The dataset is preprocessed: tokenization is performed, vocabulary dictionaries are created.
- A parametrized data loader is defined which delivers batches of token sequences with their expected target token. Basically, if we have a sequence `X` of `N` tokens, the target `y` is the next token in the script; and all that is provided in batches of a desired size.
- Definition of a RNN, which has:
  - An embedding layer.
  - An LSTM layer with parametrized layers within it.
  - A fully connected layer.
- Training of the network.
- Generation of new scripts.

### Dependencies

You should create a python environment (e.g., with [conda](https://docs.conda.io/en/latest/)) and install the dependencies listed in the [requirements.txt](requirements.txt) file.

A short summary of commands required to have all in place is the following:

```bash
conda create -n text-gen python=3.6
conda activate text-gen
conda install pytorch -c pytorch 
conda install pip
pip install -r requirements.txt
```

## Some Brief Notes on RNNs and Their Application to Language Modeling

While [Convolutional Neural Networks (CNNs)](https://en.wikipedia.org/wiki/Convolutional_neural_network) are particularly good at capturing spatial relationships, [Recursive Neural Networks (RNNs)](https://en.wikipedia.org/wiki/Recurrent_neural_network) model sequential structures very efficiently. Also, in recent years, the [Transformer](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) architecture has been shown to work remarkably well with language data -- but let's keep it aside for this small toy project.

In order to model language, the first step consists in transforming 

The vocabulary is basically a bidirectionally indexed list of words/tokens; bidirectional, because we can get from and index its text form and vice versa.

With the vocabulary defined, we can represent each word in two forms:

1. As **one-hot encoded sparse vectors** of size `(n,)`, being `n` the number of words in the vocabulary.
2. As compressed vectors from an **embedding** of size `(m,)`, with `m < n`.

In a sparse representation, a word is a vector of zeroes except in the index which corresponds to the text form in the vocabulary, where the vector element value is 1.

In a compressed representation we don't have sparse one-hot encoded vectors, but vectors of much smaller sizes that contain float values in all their cells. We can create those vector spaces in different ways; one option is to compress sparse vectors to a latent space (e.g., with an [autoencoder](https://en.wikipedia.org/wiki/Autoencoder)):

`[0, 1, 0, ..., 0] (n: vocabulary size) -> [0.2, 0.5, ..., 0.1] (m: latent word vector space)`

Typical sizes are `n = 70,000`, `m = 300`.

There are also **semantic embeddings**, which preserve the meaning relationships of words encoded as spatial relationships in the embedding spaces; thus, we can perform algebraic operations on the words:

`vector(queen) ~ vector(king) - vector(man) + vector(woman)`


I you are looking for more information on how RNNs work and Natural Language Processing (NLP), you can have a look at the following links:

- My repository [text_sentiment](https://github.com/mxagar/text_sentiment)
- [My NLP Guide](https://github.com/mxagar/nlp_guide)
- [My notes and code](https://github.com/mxagar/deep_learning_udacity) on the [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101).

## Notes on the Text Generation Application

Particular challenge: understanding the sizes of all the vectors.

- Batching sequences
- Overall definition of RNNs with LSTM units: the sizes of the different vectors, and how to deal with their reshaping.
- Hints on hyperparameter selection.


## Improvements, Next Steps

- [ ] Try different model weight initializations (e.g., for the embedding layer) to check if it is possible to improve model convergence.
- [ ] Hyperparameter tuning.

## Interesting Links

- [My notes and code](https://github.com/mxagar/computer_vision_udacity) on the [Udacity Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891).
- [My notes and code](https://github.com/mxagar/deep_learning_udacity) on the [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101).

Related interesting projects:

- [Character-level LSTM to generate text](https://github.com/mxagar/CVND_Exercises/blob/master/2_4_LSTMs/3_1.Chararacter-Level%20RNN%2C%20Exercise.ipynb), based on [a post by Andrej Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).
- Generating Bach music: [DeepBach](https://arxiv.org/pdf/1612.01010.pdf).
- Predicting seizures in intracranial EEG recordings: [American Epilepsy Society Seizure Prediction Challenge](https://www.kaggle.com/c/seizure-prediction).

## Authorship

Mikel Sagardia, 2022.  
No guarantees.

You are free to use this project, but please link it back to the original source.
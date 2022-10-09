# Text Generation Project: Writing TV Scripts

This repository contains a text generator which works with a Recurrent Neural Network (RNN) based on LSTM units. The [Seinfeld Chronicles Dataset from Kaggle](https://www.kaggle.com/datasets/thec03u5/seinfeld-chronicles) is used, which contains the complete scripts from the [Seinfield TV Show](https://en.wikipedia.org/wiki/Seinfeld).

The project is a modification of the [Character-level RNN](https://github.com/karpathy/char-rnn) [implemented by Andrej Karpathy](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). I have used materials from the [Udacity Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891), which can be obtained in their original form in [project-tv-script-generation](https://github.com/mxagar/deep-learning-v2-pytorch/tree/master/project-tv-script-generation).

If you're interested in the topic, I recommend you to read [my blog post on it](https://mikelsagardia.io/blog/text-generation-rnn.html), where I introduce Recurrent Neural Network (RNN) based on LSTM units and their application to language modeling.

Regarding the results, even though the text that the trained model is able to generate doesn't make much sense, it seems it follows the general structure that the scripts from the dataset have:

```
jerry: you know, it's the way i can do. i don't know what the hell happened.

jerry: what?

george: what about it?

elaine: i think you could be able to get out of here.

jerry: oh, i can't do anything about the guy.

jerry: what?

george:(smiling) yeah..........

george: you know, you should do the same thing.

jerry: i think i can.

jerry: oh, no, no! no. no.

jerry: i don't know.(to the phone) what do you think?

george: what?

jerry: oh, i think you're not a good friend.

...
```

... and that's with very few hours of effort and GPU training, so I I'd say it's a good starting point :sweat_smile:

Table of Contents:

- [Text Generation Project: Writing TV Scripts](#text-generation-project-writing-tv-scripts)
  - [How to Use This](#how-to-use-this)
    - [Overview of Files and Contents](#overview-of-files-and-contents)
    - [Dependencies](#dependencies)
  - [Some Brief Notes on RNNs and Their Application to Language Modeling](#some-brief-notes-on-rnns-and-their-application-to-language-modeling)
  - [Practical Notes on the Text Generation Application](#practical-notes-on-the-text-generation-application)
  - [Improvements, Next Steps](#improvements-next-steps)
  - [Interesting Links](#interesting-links)
  - [Authorship](#authorship)

## How to Use This

In order to use the model, you need to install the [dependencies](#dependencies) and execute the notebook [tv_script_generation.ipynb](tv_script_generation.ipynb), which is the main application file that defines and trains the network.

Next, I give a more detailed description on the contents and the usage.

### Overview of Files and Contents

Altogether, the project directory contains the following files and folders:

```
.
├── Instructions.md                     # Original Udacity instructions
├── README.md                           # This file
├── data/                               # Dataset folder
│   └── Seinfeld_Scripts.txt            # Dataset: scripts
├── tv_script_generation.ipynb          # Project notebook
├── helper.py                           # Utilities: load/preprocess data, etc.
├── problem_unittests.py                # Unit tests
└── requirements.txt                    # Dependencies
```

As already introduced, the notebook [tv_script_generation.ipynb](tv_script_generation.ipynb) takes care of almost everything. That file uses the following two scripts:

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

When the complete notebook is executed, several other artifacts are generated:

- A binary with the pre-processed text.
- The trained models (best and last).
- A TV script generated with the trained model.

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

While [Convolutional Neural Networks (CNNs)](https://en.wikipedia.org/wiki/Convolutional_neural_network) are particularly good at capturing spatial relationships, [Recurrent Neural Networks (RNNs)](https://en.wikipedia.org/wiki/Recurrent_neural_network) model sequential structures very efficiently. Also, in recent years, the [Transformer](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) architecture has been shown to work remarkably well with language data -- but let's keep it aside for this small toy project.

In many language modeling applications, and in the particular text generation case  explained here, we need to undertake the following general steps:

- The text needs to be **processed** as sequences of numerical vectors.
- We define **recurrent layers** which take those sequences of vectors and yield sequences of outputs.
- We take the complete or partial output sequence and we **map it to the target space**, e.g., words.

If you'd like to know more about how these steps, you should my [blog post](https://mikelsagardia.io/blog/text-generation-rnn.html) on the project.

## Practical Notes on the Text Generation Application

CONTENT.

Particular challenge: understanding the sizes of all the vectors.

- Batching sequences
- Overall definition of RNNs with LSTM units: the sizes of the different vectors, and how to deal with their reshaping.
- Hints on hyperparameter selection. [char-rnn](https://github.com/karpathy/char-rnn).

## Improvements, Next Steps

- [ ] Try different model weight initializations (e.g., for the embedding layer) to check if it is possible to improve model convergence.
- [ ] Carry out hyperparameter tuning.

## Interesting Links

- [My blog post on the project](https://mikelsagardia.io/blog/text-generation-rnn.html).
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

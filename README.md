# Introduction

In this project, I built multiple deep neural networks (simple RNN, RNN with Embedding, Bidirectional RNN, Encoder-Decoder RNN, Encoder-Decoder Bidirectional RNN with Embedding) that can function as part of an end-to-end machine translation pipeline. The pipeline accepts English text as input and returns the French translation. The pipeline consists of:

- Data Preprocessing
- Model Training
- Model Predictions


# Setup

This project requires GPU acceleration to run efficiently.

## Local Machine (Option)

If you are planning to run on a local machine, I recommend only doing this option if you have a powerful GPU meant for deep learning. For instance, [ASUS - ROG GU502GV 15.6" Gaming Laptop - Intel Core i7 - 16GB Memory - NVIDIA GeForce RTX 2060 - 1TB SSD + Optane - Brushed Metallic Black](https://www.bestbuy.com/site/asus-rog-gu502gv-15-6-gaming-laptop-intel-core-i7-16gb-memory-nvidia-geforce-rtx-2060-1tb-ssd-optane-brushed-metallic-black/6338248.p?skuId=6338248).


## Amazon Web Services (Option)

Launch a GPU EC2 instance. For instance, you can choose to launch [AWS Deep Learning AMI (Ubuntu 18.04)](https://aws.amazon.com/marketplace/pp/Amazon-Web-Services-AWS-Deep-Learning-AMI-Ubuntu-1/B07Y43P7X5) as a GPU instance.
 

## Install

- Python 3.5
- NumPy
- TensorFlow GPU 1.3.0
- Keras 2.0.9
- Jupyter
- Cython

~~~bash
conda create --name machine-translation python=3.5 numpy
conda activate machine-translation
pip install Cython
pip install tensorflow-gpu==1.3.0
pip install keras==2.0.9
pip install jupyter
cd path/to/project
jupyter notebook
~~~

# Machine Translation Pipeline

## Data Preprocessing

- Tokenization: Tokenize the words into ids
- Padding: Add padding to make all the sequences the same length

## Model Training

The models that were built and then trained include:

- Model 1: simple RNN
- Model 2: RNN with Embedding
- Model 3: Bidirectional RNN
- Model 4: Encoder-Decoder RNN
- Model 5: Custom Encoder-Decoder Bidirectional RNN with Embedding (Final Model)

For models 1 - 4, I set them to be trained over 10 epochs with a validation split of 80% for training and 20% for validation. For model 5, I set it to be trained over 30 epochs with the same validation split as models 1 - 4. I increased the epochs for training model 5, so my model correctly predicted both sentences used for testing.

## Model Predictions

Each model was trained to translate from english to french text. I included the validation accuracy as it was a good indicator to determine how well each model would do when translating from new english text it had not seen before to french text.

| Model   | Validation Accuracy |
| :-----: | :-----------------: |
| Model 1 | 61.85 |
| Model 2 | 79.47 |
| Model 3 | 67.07 |
| Model 4 | 68.23 |
| Model 5 | 95.32 |

By examining the table, we can determine **Model 5: Custom Encoder-Decoder Bidirectional RNN with Embedding (Final Model)** would perform the best when translating from new english text it had not seen before to french text. Through the testing in the project, Model 5 was most accurate.

# Future Enhancements

- Add the ability to translate from English text to other languages, such as Spanish, Japanese, etc.
- Develop a web application or mobile application
- Integrate with Audio Recognition to translate audio English text to another language, such as French, Spanish, etc

# Resources

- [Encoder-Decoder Long Short-Term Memory Networks](https://machinelearningmastery.com/encoder-decoder-long-short-term-memory-networks/)
- [A Bouquet of Sequence to Sequence Architectures for Implementing Machine Translation](https://medium.com/analytics-vidhya/a-bouquet-of-sequence-to-sequence-architectures-for-implementing-machine-translation-5d13b286df5)
- [Language Translation with RNNs](https://towardsdatascience.com/language-translation-with-rnns-d84d43b40571)
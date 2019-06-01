# Aspect extraction from product reviews with Tensorflow
* Must uninstall python 3.7 and install python 3.6
This repo has multiple sequential models for aspect extraction from product reviews.

## Task

Given a sentence, the task is to extract aspects. Here is an example

```
I like the battery life of this phone"

Converting this sentence to IOB would look like this -

I O
like O
the O
battery B-A
life I-A
of O
this O
phone O

```


## Model

Similar to [Ma and Hovy](https://arxiv.org/pdf/1603.01354.pdf).

- concatenate final states of a bi-lstm on character embeddings to get a character-based representation of each word
- concatenate this representation to a standard word vector representation (GloVe here)
- run a bi-lstm on each sentence to extract contextual representation of each word
- decode with a linear chain CRF

Similar to [Collobert et al.] (http://ronan.collobert.com/pub/matos/2011_nlp_jmlr.pdf)

- form a window around the word to tag
- apply MLP on that window
- obtain logits
- apply viterbi (CRF) for sequence tagging

Similar to [Poria et al.](https://www.sciencedirect.com/science/article/pii/S0950705116301721)

- form a window around the word to tag
- apply CNN on that window
- apply maxpool on that window (Caution: different from global maxpool)
- obtain logits
- apply CRF for sequence tagging

## XML to IOB

```
python xmlToIOB.py
```

## Details

Download Glove embeddings (GloVe: http://nlp.stanford.edu/data/glove.840B.300d.zip )

1. [DO NOT MISS THIS STEP] Build vocab from the data and extract trimmed glove vectors according to the config in `model/config.py`.

```
python build_data.py
```

After 'build_data.py'

<a href="https://imgur.com/0Ky7SjU"><img src="https://i.imgur.com/0Ky7SjU.png" title="source: imgur.com" /></a>

 (1) chars: characters.. of what? <br>
 (2) glove trimmed npz: numpy array of trimmed glove vocabs <br>
 (3) tags: IOB tags of training data<br>
 (4) words: words of training data<br>


2. Train the model with

```
python train.py
```


3. Evaluate and interact with the model with
```
python evaluate.py
```


Data iterators and utils are in `model/data_utils.py` and the model with training/test procedures is in `model/aspect_model.py`


## Training Data


The training data must be in the following format (identical to the CoNLL2003 dataset).

-----

The CoNLL-2003 shared task data files contain four columns separated by a single space. Each word has been put on a separate line and there is an empty line after each sentence. The first item on each line is a word, the second a part-of-speech (POS) tag, the third a syntactic chunk tag and the fourth the named entity tag. The chunk tags and the named entity tags have the format I-TYPE which means that the word is inside a phrase of type TYPE. Only if two phrases of the same type immediately follow each other, the first word of the second phrase will have tag B-TYPE to show that it starts a new phrase. A word with tag O is not part of a phrase. Here is an example:

| Word | POS tag | Syntactic chunk tag | Named entity tag |
| ---- | ---- | ---- | ---- |
|   U.N.|NNP|I-NP|I-ORG |
|official|NN|I-NP|O|
|Ekeus|NNP|I-NP|I-PER| 
|heads|VBZ|I-VP|O| 
|for|IN|I-PP|O|
|Baghdad|NNP|I-NP|I-LOC| 
|.|.|O|O| 
   
-----



A default test file is provided to help you getting started.


```
The	O
duck	B-A
confit	I-A
is	O
always	O
amazing	O
and	O
the	O
foie	B-A
gras	I-A
terrine	I-A
with	I-A
figs	I-A
was	O
out	O
of	O
this	O
world	O

The	O
wine	B-A
list	I-A
is	O
interesting	O
and	O
has	O
many	O
good	O
values	O
```


Once you have produced your data files, change the parameters in `config.py` like

```
# dataset
filename_train = "data/ABSA16_Restaurants_Train_SB1_v2_mod.iob"
filename_dev = "data/EN_REST_SB1_TEST_2016_mod.iob"
filename_test = "data/EN_REST_SB1_TEST_2016_mod.iob"
```

## Result

Chunk based evaluation

```
Laptop 2014 -> F1 - 79.93

Restaurant 2014 -> F1 - 84.22
```
Partial matching based evaluation
```
Laptop 2014 -> F1 - 86.84
Restaurant 2014 -> F1 - 88.42
```
## Citation


Poria, S., Cambria, E. and Gelbukh, A., 2016. Aspect extraction for opinion mining with a deep convolutional neural network. Knowledge-Based Systems, 108, pp.42-49.


## License

This project is licensed under the terms of the apache 2.0 license (as Tensorflow and derivatives). If used for research, citation would be appreciated.


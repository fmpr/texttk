==============================================
texttk -- Text Preprocessing in Python
==============================================

texttk is a Python library for *text preprocessing* of large corpora, that can be used for *topic modelling*, *text classification*, *document clustering*, *information retrieval*, etc.

Features
---------

* Removes stopwords, punctuation, HTML tags, accents, rare words, very frequent words, etc.
* Applies stemmer or lemmatizer
* Finds the most relevant bigrams
* Performs named entity recognition
* Tokenizes text
* Generates bag-of-words representations
* Identifies and joins compound words
* Splits documents in sentences

Default preprocessing pipeline
---------

* unescape characters
* remove HTML tags
* strip accents
* join URLs
* join compound words
* space out punctuation
* split documents in sentences
* tag sentences using Stanford NER
* merge named entities as single tokens
* remove punctuation
* convert to lowercase
* find top-100 bigrams using chi-squared
* merge bigrams as single tokens
* tokenize documents
* remove stopwords
* apply stemmer/lemmatizer
* remove rare words
* remove very frequent words
* convert corpus to bag-of-words format

Basic Usage
------------

Basic preprocessing of a corpus and transformation to bag-of-words format::

	import texttk
    corpus = ... # a list of strings
    tp = texttk.TextPreprocesser(decode_error='strict', strip_accents='unicode', ignore_list=[], \
				lowercase=True, remove_html=True, join_urls=True, use_bigrams=True, \
				use_ner=True, stanford_ner_path="<path_here>", use_lemmatizer=False, \
				max_df=0.95, min_df=1, max_features=None)
    corpus = tp.preprocess_corpus(corpus)
    transformed_corpus, vocab = tp.convert_to_bag_of_words(corpus)

Installation
------------

This software depends on `nltk`, `sklearn` and `HTMLParser`.
You must have them installed prior to installing `texttk`.
If you wish to use NER, `texttk` also requires access to the Stanford NER Java library.

The simple way to install `texttk` is::

    python setup.py install

`texttk` is open source software released under the `GNU LGPL license <http://www.gnu.org/licenses/lgpl.html>`_.
Copyright (c) 2016-now Filipe Rodrigues


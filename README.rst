==============================================
texttk -- Text Preprocessing in Python
==============================================

texttk is a Python library for *text preprocessing* of large corpora, that can be used for *topic modelling*, *text classification*, *information retrieval*, etc.

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
* remove html
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

Installation
------------

This software depends on `ntlk`, `sklearn` and `HTMLParser`.
You must have them installed prior to installing `texttk`.
If you wish to use NER, `texttk` also requires access to the Stanford NER Java library.

The simple way to install `texttk` is::

    python setup.py install

Gensim is open source software released under the `GNU LGPL license <http://www.gnu.org/licenses/lgpl.html>`_.
Copyright (c) 2016-now Filipe Rodrigues


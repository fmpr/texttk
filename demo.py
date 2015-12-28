#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import texttk
import csv, codecs

# load data from the csv file
def read_csv_utf8(filename, delimiter=',', is_header=True):
	rows, header = [], ""
	rownum = 0
	corpus = []
	with open(filename, "r") as f:
		reader = csv.reader(f, delimiter=delimiter)
		for row in reader:
			if is_header and rownum == 0: header = row
			elif row: corpus.append(row[14].decode("utf-8"))
			rownum += 1
			if rownum > 1000:
				break
	return corpus

# load corpus from directory
def read_utf8_corpus(self, directory):
	print "reading utf8 corpus from %s" % (directory,)
	corpus = []
	for fname in os.listdir(directory):
		f = codecs.open(directory + fname, "r", "utf-8")
		f.readline()
		doc = ""
		for line in f:
			splt = line.split(",")
			doc += splt[0] + " " + splt[1] + " "
		f.close()
		corpus.append(doc)
		if len(corpus) > 10:
			break
	return corpus

def demo():
	# load corpus
	#corpus = read_utf8_corpus("/Users/fmpr/Downloads/events_search_results/query_search_results/") 
	corpus = read_csv_utf8("/Users/fmpr/code/ioracle/data/sg_events_join_venues_26052015_whole_island_utf8.csv")
	
	# debug text preprocessing
	for d in xrange(len(corpus)):
		fw = codecs.open("debug/%d_original.txt" % (d,), "w", "utf-8")
		fw.write(corpus[d])
		fw.close()

	ignore_list = ["singapor", "sg", "expo", "sale", "event", "exhibit", "fair", "com", "asia"]
	stanford_ner_path = "/Users/fmpr/code/texttk/stanford-ner-2015-12-09/"
	tp = texttk.TextPreprocesser(decode_error='strict', strip_accents='unicode', ignore_list=ignore_list, lowercase=True, \
						remove_html=True, join_urls=True, use_bigrams=True, use_ner=True, stanford_ner_path=stanford_ner_path, \
						use_lemmatizer=False, max_df=0.95, min_df=1, max_features=None)

	corpus = tp.preprocess_corpus(corpus)

	# debug text preprocessing
	for d in xrange(len(corpus)):
		fw = codecs.open("debug/%d_cleaned.txt" % (d,), "w", "utf-8")
		fw.write(corpus[d])
		fw.close()

	dtm, vocab = tp.convert_to_bag_of_words(corpus)

	# debug text preprocessing
	for d in xrange(len(corpus)):
		fw = codecs.open("debug/%d_preproc.txt" % (d,), "w", "utf-8")
		for n in dtm[d].nonzero()[1]:
			fw.write(vocab[n] + " ")
		fw.close()

if __name__ == "__main__":
    demo()


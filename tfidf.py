#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
author: sanja7s
'''

# 
import sys, io, json, os
import numpy as np
import scipy
from collections import defaultdict
# CountVectorizer counts the number of times a term occurs in the document
from sklearn.feature_extraction.text import CountVectorizer
# TfidfTransformer will use output from CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import itertools
# to parse xml at some points
import xml.etree.cElementTree as ET
import re, random
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
# to prune date articles
from dateutil.parser import *
from sklearn.preprocessing import normalize



#####################################################################################################################

NON_STOPWORD_LIMIT = 200
STEMMER = SnowballStemmer("english", ignore_stopwords=True )

def stem_article( article ):
	stemmed_tokens = []
	for token in wordpunct_tokenize( article ):
		token = token.lower()
		if token.isalpha():
			stemmed_tokens.append( STEMMER.stem( token ) )
	return ' '.join( stemmed_tokens )

#####################################################################################################################
# TF-IDF
#####################################################################################################################
# Code that calculates TF-IDF is adapted from the blogpost by Christian S. Perone
# http://blog.christianperone.com/?p=1589
# The code uses scikits.learn Python module for machine learning http://scikit-learn.sourceforge.net/stable/

# 1 Normalize TF-IDF (Gabrilovich)
# cnt_articles is the number of articles we kept after preprocessign and cleaning the Wikipedia data
# article_ids is a dictionary with cnt_articles items: Wiki article ids, indexed by our internal ids
# train_set_dict is a dictionary with cnt_articles items: cleaned texts of Wiki articles, indexed by our internal ids
def tfidf_normalize(articles_with_id):
	global NON_STOPWORD_LIMIT
	stemmed_articles_with_id = [ (aid, stem_article( article )) for (aid, article) in articles_with_id ]
	stemmed_articles = [ article for (aid, article) in stemmed_articles_with_id ]
	#test_set = train_set
	# instantiate vectorizer with English language, using stopwords and set min_df, max_df parameters and the tokenizer
	vectorizer = CountVectorizer(stop_words='english', min_df=3, max_df=0.1, token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b')
	# by appling the vectorizer instance to the train set
	# it will create a vocabulary from all the words that appear in at least min_df and in no more than max_df
	# documents in the train_set
	vectorizer.fit_transform(stemmed_articles)
	# vectorizer transform will apply the vocabulary from the train set to the test set. In my case,
	# they are the same set: whole Wikipedia.
	# this means that each article will get representation based on the words from the vocabulary and
	# their TF-IDF values in the Scipy sparse output matricx
	freq_term_matrix = vectorizer.transform(stemmed_articles)
	long_articles_with_id = []
	assert freq_term_matrix.shape[0] == len(articles_with_id)
	for (i, article_with_id) in zip( xrange(freq_term_matrix.shape[0]), stemmed_articles_with_id ): 
		row = freq_term_matrix.getrow(i)
		if row.getnnz() >= NON_STOPWORD_LIMIT:
			long_articles_with_id.append( article_with_id )

	long_articles = [ article for (aid, article) in long_articles_with_id ]

	vectorizer = CountVectorizer(stop_words='english', min_df=3, max_df=0.1, token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b')
	vectorizer.fit_transform(long_articles)

	freq_term_matrix = vectorizer.transform(long_articles)

	# Gabrilovich says that they threshold TF on 3 (remove word-article association if that word
	# does not appear at least 3 times in that single article
	#freq_term_matrix.data *= freq_term_matrix.data>=3
	#freq_term_matrix.eliminate_zeros() # I think this is not necessary...
	# this is a log transformation as applied in (Gabrilovich, 2009), i.e., that is
	# how he defines TF values. In case of TF = 0, this shall not affect such value
	# freq_term_matrix.data = 1 + np.log( freq_term_matrix.data )
	# instantiate tfidf trnasformer
	tfidf = TfidfTransformer(norm=None, smooth_idf = False, sublinear_tf = True)
	# tfidf uses the freq_term_matrix to calculate IDF values for each word (element of the vocabulary)
	tfidf.fit(freq_term_matrix)
	# finally, tfidf will calculate TFIDF values with transform()
	tf_idf_matrix = tfidf.transform(freq_term_matrix)
	# tf_idf_matrix.data = np.log(np.log(tf_idf_matrix.data))
	tf_idf_matrix = normalize(tf_idf_matrix, norm="l2", axis = 0, copy=False)
	# now we put our matrix to CSC format (as it helps with accessing columns for inversing the vectors to
	# words' concept vectors)
	tf_idf_matrix = tf_idf_matrix.tocsc() 
	# we need vocabulary_ to be accessible by the index of the word so we inverse the keys and values of the
	#dictionary and put them to new dictionary word_index
	word_index = dict((v, k) for k, v in vectorizer.vocabulary_.iteritems())
	M, N = tf_idf_matrix.shape
	print "Articles: ", M
	print "Words: ", N
	return tf_idf_matrix, word_index, long_articles_with_id


# 2 Raw TF-IDF values (Hieu)
# cnt_articles is the number of articles we kept after preprocessign and cleaning the Wikipedia data
# article_ids is a dictionary with cnt_articles items: Wiki article ids, indexed by our internal ids
# train_set_dict is a dictionary with cnt_articles items: cleaned texts of Wiki articles, indexed by our internal ids
def tfidf_raw(cnt_articles, article_ids, train_set_dict):
	# use the whole (Wiki) set as both the train and test set
	train_set = train_set_dict.values()
	test_set = train_set
	#train_set = ("The sky is blue.", "The sun is bright.")
	#test_set = ("The sun in the sky is bright.","We can see the shining sun, the bright sun.")
	vectorizer = CountVectorizer(stop_words='english')
	# instantiate vectorizer with English language, using stopwords and set min_df, max_df parameters and the tokenizer
	# vectorizer = CountVectorizer(stop_words='english', min_df=2, max_df=0.7, token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b')
	# by appling the vectorizer instance to the train set
	# it will create a vocabulary from all the words that appear in at least min_df and in no more than max_df
	# documents in the train_set
	vectorizer.fit_transform(train_set)
	#print "Vocabulary:", vectorizer.vocabulary_
	# vectorizer transform will apply the vocabulary from the train set to the test set. In my case,
	# they are the same set: whole Wikipedia.
	# this means that each article will get representation based on the words from the vocabulary and
	# their TF-IDF values in the Scipy sparse output matricx
	freq_term_matrix = vectorizer.transform(test_set)
	print freq_term_matrix.todense()
	# this is a log transformation as applied in (Gabrilovich, 2009), i.e., that is
	# how he defines TF values. In case of TF = 0, this shall not affect such value
	# freq_term_matrix.data = 1 + np.log( freq_term_matrix.data )
	# instantiate tfidf trnasformer
	tfidf = TfidfTransformer(norm=None, smooth_idf = False, sublinear_tf = True)
	#print tfidf
	# tfidf uses the freq_term_matrix to calculate IDF values for each word (element of the vocabulary)
	tfidf.fit(freq_term_matrix)
	#print tfidf.idf_
	# finally, tfidf will calculate TFIDF values with transform()
	tf_idf_matrix = tfidf.transform(freq_term_matrix)
	print
	#print tf_idf_matrix.todense()
	print
	# now we put our matrix to CSC format (as it helps with accessing columns for inversing the vectors to
	# words' concept vectors)
	CSC_matrix = tf_idf_matrix.tocsc() 
	CSC_matrix = normalize(CSC_matrix, norm="l2", axis = 0, copy=False)
	# we need vocabulary_ to be accessible by the index of the word so we inverse the keys and values of the
	#dictionary and put them to new dictionary word_index
	word_index = dict((v, k) for k, v in vectorizer.vocabulary_.iteritems())
	print word_index
	M, N = CSC_matrix.shape
	print "Articles: ", M
	print "Words: ", N
	return M, N, CSC_matrix, word_index
#####################################################################################################################

#####################################################################################################################
# OUTPUT
#####################################################################################################################
# save desired output for the next step (input to MongoDB) in JSON file, format:
# CV = word, article_id1:tfidf1 ... article_idN:tfidfN
# go through columns and print the word and the indices (= article_id) and the column data (= tfidf)

# 1 save with sliding window pruning (Gabrilovich)
def save_CV_with_sliding_window_pruning(m, fn, word_index, article_ids, window=100, drop_pct=0.5):
	M, N = m.shape
	CNT = 0
	chck_pruning = 0
	TEST_WORD = "new"
	with io.open(fn, 'w', encoding='utf-8') as f:
		for j in range(N):
			CNT += 1
			CV_dict = {}
			the_word = word_index[j]
			CV_dict['_id'] = the_word
			CV_dict['CV'] = []
			col = m.getcol(j)
			if len(col.data) == 0:
				continue
			highest_scoring_concept = max(col.data)
			highest_scoring_concept_pct = highest_scoring_concept * drop_pct/100.0
			remembered_tfidf = highest_scoring_concept
			remembered_id = 0
			k = 0
			for (idx, value) in sorted(zip(col.indices, col.data), key = lambda t: t[1], reverse=True):
				k += 1
				tfidf_dict = {}
				tfidf_dict[str(idx)] = (str(value),str(article_ids[idx]))
				CV_dict['CV'].append(tfidf_dict)
				if the_word == TEST_WORD:
					print value, remembered_tfidf, highest_scoring_concept_pct, remembered_tfidf - value
				#print col.data
				if k >= window:
					if remembered_tfidf - value < highest_scoring_concept_pct:
						if the_word == TEST_WORD:
							print "PRUNED ", value, remembered_tfidf, highest_scoring_concept_pct, remembered_tfidf - value
						chck_pruning += (len(col.data) - k)
						break
					else:
						remembered_id += 1
						remembered_tfidf = np.sort(col.data)[::-1][remembered_id]
						# test
						#if the_word == "new":
							#print highest_scoring_concept_pct, remembered_id, remembered_tfidf, value
							#print np.sort(col.data)[::-1]
			if the_word == TEST_WORD:
				print CV_dict
				print np.sort(col.data)[::-1]
			f.write(unicode(json.dumps(CV_dict, ensure_ascii=False)) + '\n')
			#if CNT % 100 == 0:
				#print CNT, the_word, highest_scoring_concept, value, len(col.data), highest_scoring_concept_pct
	print "ESA_1 Pruned terms total: ", chck_pruning


# 2 save with threshold pruning (Hieu)
def save_CV_with_threshold_pruning(m, fn, word_index, article_ids, threshold=12):
	M, N = m.shape
	CNT = 0
	chck_pruning = 0
	with io.open(fn, 'w', encoding='utf-8') as f:
		for j in range(N):
			CNT += 1
			CV_dict = {}
			the_word = word_index[j]
			CV_dict['_id'] = the_word
			CV_dict['CV'] = []
			col = m.getcol(j)
			highest_scoring_concept = max(col.data)
			selected_terms = 0
			for (idx, value) in sorted(zip(col.indices, col.data), key = lambda t: t[1], reverse=True):
				if value >= threshold:
					tfidf_dict = {}
					tfidf_dict[str(idx)] = (str(value),str(article_ids[idx]))
					CV_dict['CV'].append(tfidf_dict)
					selected_terms += 1
				else:
					chck_pruning += len(col.data) - selected_terms
					break
			f.write(unicode(json.dumps(CV_dict, ensure_ascii=False)) + '\n')
			if CNT % 10000 == 0:
				print CNT, the_word, highest_scoring_concept, value
	print "ESA_2 Pruned terms total: ", chck_pruning



# 3 save without any pruning (for smaller datasets)
def save_CV_all(m, fn, word_index, article_ids):
	M, N = m.shape
	CNT = 0
	with io.open(fn, 'w', encoding='utf-8') as f:
		for j in range(N):
			CNT += 1
			CV_dict = {}
			the_word = word_index[j]
			CV_dict['_id'] = the_word
			CV_dict['CV'] = []
			col = m.getcol(j)
			if len(col.data) == 0:
				continue
			highest_scoring_concept = max(col.data)
			selected_terms = 0
			for (idx, value) in sorted(zip(col.indices, col.data), key = lambda t: t[1], reverse=True):
				tfidf_dict = {}
				tfidf_dict[str(idx)] = (str(value),str(article_ids[idx]))
				CV_dict['CV'].append(tfidf_dict)
			f.write(unicode(json.dumps(CV_dict, ensure_ascii=False)) + '\n')
			if CNT % 10000 == 0:
				print CNT, the_word, highest_scoring_concept, value
	print "ESA Saved without pruning terms. "


	#####################################################################################################################





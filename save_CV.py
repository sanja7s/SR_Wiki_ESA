
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
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# to prune date articles
from dateutil.parser import *
from sklearn.preprocessing import normalize


'''
# 3 save without any pruning (for smaller datasets)
def save_CV_all(tf_idf_matrix, word_index, long_articles, F_OUT_CV):
	M, N = tf_idf_matrix.shape
	CNT = 0
	with io.open(F_OUT_CV, 'w', encoding='utf-8') as f:
		for j in range(N):
			CNT += 1
			CV_dict = {}
			the_word = word_index[j]
			CV_dict['_id'] = the_word
			CV_dict['CV'] = []
			col = tf_idf_matrix.getcol(j)
			if len(col.data) == 0:
				continue
			highest_scoring_concept = max(col.data)
			selected_terms = 0
			for (row, value) in sorted(zip(col.indices, col.data), key = lambda t: t[1], reverse=True):
				tfidf_dict = {}
				tfidf_dict[str(row)] = (str(value),str(long_articles[row][0]))
				CV_dict['CV'].append(tfidf_dict)
			f.write(unicode(json.dumps(CV_dict, ensure_ascii=False)) + '\n')
			if CNT % 10000 == 0:
				print CNT, the_word, highest_scoring_concept, value
	print "ESA Saved without pruning terms. "
'''

	# 2 save with threshold pruning (Hieu)
def save_CV_all(tf_idf_matrix, word_index, long_articles, F_OUT_CV, threshold = 0.005, window = 100):
	M, N = tf_idf_matrix.shape
	CNT = 0
	chck_pruning = 0
	TEST_WORD = "new"
	with io.open(F_OUT_CV, 'w', encoding='utf-8') as f:
		for j in range(N):
			CNT += 1
			CV_dict = {}
			the_word = word_index[j]
			CV_dict['_id'] = the_word
			CV_dict['CV'] = []
			col = tf_idf_matrix.getcol(j)
			if len(col.data) == 0:
				continue
			highest_scoring_concept = max(col.data)
			highest_scoring_concept_pct = highest_scoring_concept * threshold
			remembered_tfidf = highest_scoring_concept
			remembered_id = 0
			k = 0
			for (row, value) in sorted(zip(col.indices, col.data), key = lambda t: t[1], reverse=True):
				k += 1
				tfidf_dict = {}
				tfidf_dict[str(row)] = (str(value),str(long_articles[row][0]))
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
				#print CNT, the_word, highest_scoring
#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
author: sanja7s

This code calculates a knowledge base that can be used to calculate semantic relatendess (SR) from Wikipedia data
according to Explicit Semantic Analysis (ESA) algorithm (Gabrilovich et al. 2007).


Theory:
Basic idea of ESA is that the more similar the words are, they will be found in more common Wikipedia articles.
And the importance of a word in an article is also taken into account; determined using TF-IDF (term frequency -
inverse document frequency) algorithm. Thus the ESA algorithm takes (latest) Wikipedia dump and determines
importance of different words (terms) in different articles using TF-IDF algorithm. An intermediary result is a set
of vectors for each Wikipedia article (concept) with corresponding TF-IDF values for words that are found relevant
in this article. Then the inverse vector for each word is calculated (containing its TF-IDF values in different
articles), and such a vector serves to calculate SR value of the words. 


Implementation:
Preprocessing: Wikipedia data comes in xml format and can be quite messy. Thus we need to clean it (preprocess).
The preprocessing step is described below separately.
TF-IDF calculation: We calculate TF-IDF for the non-stopwords words in all the articles. We use sklearn
TfidfTransformer that builds a vocabulary omitting English stop_words (experimenting with minimum document frequency
for a word, and maximum percent of articles in which the word appears -- more details on this under Parameters).
TfidfTransformer outputs a sparse scipy matrix that has words as columns, articles as rows, and corresponding TF-IDF
values as elements. The sparce scipy matrix is well suited for our case since each article contains a small percent
of the all words, so the resulting TF-IDF matrix is quite sparse. From such a TF-IDF matrix, we obtain a concept
vector (CV) for each word, by simply accessing the columns of the matrix.  
Output: The final output of this code pipeline is a text file formated as a set of single-line json objects that is
suitable to be imported to MongoDB, where our knowledge base is to be stored (note: Mongo limit of a single json
object is 16MB). Each josn line is a word CV (concept vector).

Articles #:  1829625
Words #:  1294606


Preprocessing:
Wikipedia xml dump that we use is the English dump with pages and articles enwiki-20150304-pages-articles.xml.bz2.
However, there are Templates to be expanded (that can get problematic and lead to recursion issues). Then there are
redirections to be resolved. Then we also need to extract the pages with more then 5 inlinks and outlinks.
Additional cleaning of Wikipedia data for our purpose involves:
      pruning disambiguation pages ("may refer to:")
      pruning categories (if used wikiextractor by Attardi, that is done by his code)
      pruning too short articles (< 100 non-stopwords)
      pruning rare words (< N articles) (min_df)
      pruning too frequent (= corpus specific stopwords) present in percent articles (max_df)
      pruning dates (4 May, 2011, 1987)

1 There is a code provided by Gabrilovich wikiprep that is meant exactly for the purpose of ESA.
2 Another version of Wikipedia preprocessing code is by Prof. Attardi and serves more general purpose, wikiextractor.


Parameters:
- minimum document frequency for a word (min_df = 3 (Gabrilovich), min_df = 8 (Hieu) )
- maximum percent of articles in which the word appears (max_df = 0.7 (suggested in the article about TF-IDF), max_df = 0.1 (Hieu))
- default tokenizer is token_pattern=u'(?u)\b\w\w+\b'  (note: \w == [a-zA-Z0-9_])
We set token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'


Pruning the results:
1 ESA method normalizes the data. 
(tfidf = TfidfTransformer(norm=None, sublinear_tf=True), freq_term_matrix.data = 1 + np.log( freq_term_matrix.data ) )
- prune the vectors, instead, when using method by Gabrilovich (that means taking a sliding window of 100 on a sorted
concept vector by Tf-IDF, and cut when the differnce between the first and last element in the window is higher than
5% from the highest TF-IDF value in this concept vector <=> sudden drop)

2 Hieu's method (Hieu et al. 2013) is not normalized: no log transformation, just use raw TF values. 
If we want to compare words based on similar concepts, then we probably also want those to have similar TF-IDF scales. 
Ok, article lengths also represent to us their importance in "general" in some way. perhaps :)
- threshold the TF-IDF values with 12 based on (Hieu et al. 2013)



References
----------
Gabrilovich, Evgeniy, and Shaul Markovitch. "Computing Semantic Relatedness Using Wikipedia-based Explicit Semantic
Analysis." IJCAI. Vol. 7. 2007.

Gabrilovich, Evgeniy, and Shaul Markovitch. "Wikipedia-based semantic interpretation for natural language processing."
Journal of Artificial Intelligence Research (2009): 443-498.

Hieu, Nguyen Trung, Mario Di Francesco, and Antti Ylä-Jääski. "Extracting knowledge from wikipedia articles through
distributed semantic analysis." Proceedings of the 13th International Conference on Knowledge Management and Knowledge
Technologies. ACM, 2013.
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
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
# to prune date articles
from dateutil.parser import *
from sklearn.preprocessing import normalize
import codecs
import mysql.connector
from mysql.connector import errorcode

'''
#####################################################################################################################
# READ IN
#####################################################################################################################

# this code, as well as the category list is taken directly from Catagay Calli's ESA code
# https://github.com/faraday/wikiprep-esa/blob/master/scanData.py
def read_stop_cats(f_in = "Gabrliovich_preprocessed/stop_categories_cagatay_calli"):
	catList = []
	try:
		f = open(f_in,'r')
		for line in f.readlines():
			strId = line.split('\t')[0]
			if strId:
				catList.append((strId))
		f.close()
	except:
		print 'Stop categories cannot be read!'
		sys.exit(1)
	return frozenset(catList)


#####################################################################################################################
# test directly Gabrilovich articles only

def read_selected_Gab(f_in = "Gabrliovich_preprocessed/selected.txt"):
	sel_articles = []
	with open(f_in,'r') as input_file:
		for line in input_file:
			sel_articles.append(int(line))
	return sel_articles
#####################################################################################################################

# we store here our additionally cleaned dataset from the preprocessed hgw xml 
# to outfile as na optional step
def process_keywords(selected, aid, text, title, cnt_articles, article_ids, train_set_dict, outfile):
	# use nltk to find english stopwords and tokenize the text
	# while I need to join the tokens later for the purpose of TfIDFTokenizer, a good thing is that now I still clean the stopwords, 
	# which, I think, speeds up the CountVectorizer tokenizer later
	stopset = set(stopwords.words('english'))
	tokens=word_tokenize(text.decode('utf-8'))
    	tokens = [w for w in tokens if not w in stopset]
    # to really do all as Gabrilovich, we remove finally also articles with less than 100 keywords
	if len(tokens) > 100:
		train_set_dict[cnt_articles] = ' '.join(tokens)
		article_ids[cnt_articles] = int(aid)
		if cnt_articles % 10000 == 0:
			print cnt_articles, aid, title, len(tokens)
		if not selected:
			outfile.write(str(aid) + '\t' + str(cnt_articles) + '\t' + str(title) + '\n')
		cnt_articles += 1
	return cnt_articles

# extract page properties or clean the page text from buf
def process_page_OLD(selected, buf, cnt_articles, article_ids, train_set_dict, outfile, asl, STOP_CATS, articles_selected):
	# prune stop cat given in the file by Cagatay Calli
	cat = re.search(r'<categories>(.*)</categories>', buf, re.DOTALL)
	categories = cat.group(1) if cat else None
	if categories:
		cat_list = frozenset(categories.split(' '))
	else:
		return cnt_articles
	if not cat_list:
		return cnt_articles
	if cat_list.intersection(STOP_CATS):
		print "Filtered stopcat", cat_list.intersection(STOP_CATS)
		return cnt_articles
	# prune all the unecessary pages (articles based on type)
	if "This is a disambiguation page" in buf:
		#print "disambiguation"
		return cnt_articles
	if "<title>Category:" in buf:
		#print "category"
		return cnt_articles
	if "<title>List of" in buf:
		return cnt_articles
	if "<title>Timeline of" in buf:
		return cnt_articles 
	if "<title>Template:" in buf:
		return cnt_articles
	if "<title>Index of:" in buf:
		return cnt_articles 
	page = ET.fromstring(buf)
	aid = page.attrib['id']
	outlinks = page.attrib['outlinks']
	if page.attrib['stub'] == "1":
		return cnt_articles
	if int(outlinks) < 5:
		return cnt_articles
	if asl[int(aid)] < 5:
		return cnt_articles
	t = re.search(r'<title>(.*)</title>', buf, re.DOTALL)
	title = t.group(1) if t else None
	if len(title) <= 1:
		return cnt_articles
	reOtherNamespace = re.compile("^(User|Wikipedia|File|MediaWiki|Template|Help|Category|Portal|Book|Talk|Special|Media|WP|User talk|Wikipedia talk|File talk|MediaWiki talk|Template talk|Help talk|Category talk|Portal talk):.+",re.DOTALL)
	if reOtherNamespace.match(title):
		print "Namespace filtered", title
		return
	# if title of the article qualifies as date, prune it, too
	try:
		parse(title)
		#print title
		return cnt_articles
	except:
		ValueError or TypeError
	# double check for disambiguation since Wiki is not clean
	if "(disambiguation)" in title:
		return cnt_articles
	# also check for numbers
	if "(number)" in title:
		return cnt_articles
	if not int(aid) in articles_selected:
		outfile.write(str(aid) + '\t' + str(cnt_articles) + '\t' + str(title) + '\n')
		return cnt_articles
	# finally, if page passed the pruning phase, extract only its text
	m = re.search(r'<text>(.*)</text>', buf, re.DOTALL)
	text = m.group(1) if m else None
    # I choose not to care about headers and other xml markup that might be left
	text = re.sub(r'<(.*)>', '', text, re.DOTALL)
	# for the next step, I need one-line articles, so we replace new lines with spaces
	text = text.replace('\n', ' ')
	# finally process the text for non-stopwords (conceptually that code could have been here, but is cleaner this way)
	cnt_articles = process_keywords(selected, aid, text, title, cnt_articles, article_ids, train_set_dict, outfile)
	return cnt_articles

'''

def read_in_STOP_CATS(f_n = "/media/sscepano/Data/Wiki2015/STOPCAT/STOP_CATS.txt"):
	s = []
	f = open(f_n, "r")
	for line in f:
			s.append(line.rstrip().lower())
	return s

def connect_2_db():
	try:
		cnx = mysql.connector.connect(user='test', password='test',
	                              host='127.0.0.1',
	                              database='wiki_category_links')
	except mysql.connector.Error as err:
		if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
			print("Something is wrong with your user name or password")
		elif err.errno == errorcode.ER_BAD_DB_ERROR:
			print("Database does not exist")
		else:
			print(err)
	return cnx


def articles_selected(aid):
	global cnx
	global STOP_CATS
	cursor = cnx.cursor(buffered=True)
	cursor.execute("SELECT * FROM categorylinks where cl_from = " + str(aid))

	row = cursor.fetchone()
	while row is not None:
		#print(row)
		cat = row[1].lower()
		#print cat
		for el in STOP_CATS:
			if el in cat:
				return False
		row = cursor.fetchone()

	return True

cnx = connect_2_db()
STOP_CATS = read_in_STOP_CATS()
TITLE_WEIGHT = 4

from lxml import etree
parser = etree.XMLParser(recover=True, encoding="utf-8")

STEMMER = SnowballStemmer("english", ignore_stopwords=True )

def stem_article( article ):
	stemmed_tokens = []
	for token in wordpunct_tokenize( article ):
		token = token.lower()
		if token.isalpha():
			stemmed_tokens.append( STEMMER.stem( token ) )
	return ' '.join( stemmed_tokens )

def process_page(buf, articles, output_file):
	global parser
	global TITLE_WEIGHT
	page = etree.fromstring(buf, parser=parser)
	aid = page.attrib['id']
	title = page.attrib['title']
	if "(number)" in title.lower():
		return
	# TODO implement id check using sqlcategories
	if  articles_selected(aid):
		# finally, if page passed the pruning phase, extract only its text
		text = stem_article( page.text.replace('\n', ' ') )
		# finally process the text for non-stopwords (conceptually that code could have been here, but is cleaner this way)
		text = text + (title + ' ') * TITLE_WEIGHT
		# print aid, title
		output_file.write(str(aid) + '\t' + title + '\n')
		#
		articles.append((int(aid), text))
	return

#####################################################################################################################

# 2 Read in the data preprocessed by wikiextractor by Prof. Attardi 
# http://medialab.di.unipi.it/wiki/Wikipedia_Extractor
def read_in_wikiextractor_output(f_in, f_out):
	# count how many articles (i.e., lines) are read
	cnt_all_articles = 0
	articles = []

	output_file = codecs.open(f_out,'w',encoding='utf8')
	output_file.write('_id' + '\t' + 'our_id' + '\t' + 'title' + '\n')

	inputbuffer = ''
	with codecs.open(f_in,'r', encoding='utf8') as input_file:
		# the code loops through the input, forms pages when they are found, and sends them to process_page for further steps
	    append = False
	    for line in input_file:	
			if '<doc id=' in line:
				inputbuffer = line
				append = True
	  		elif '</doc>' in line:
				inputbuffer += line
				cnt_all_articles += 1
				if cnt_all_articles % 100 == 0:
					print cnt_all_articles
	 			append = False
				try:
					process_page(inputbuffer, articles, output_file)
				except TypeError as e:
					print e
				inputbuffer = None
				del inputbuffer #probably redundant...
			elif append:
				inputbuffer += line

	#outfile.close()
	print "INSERTED articles: ", len(articles), "ALL READ articles: ", cnt_all_articles
	return articles
#####################################################################################################################



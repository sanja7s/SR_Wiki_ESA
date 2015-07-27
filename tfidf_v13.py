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

import io, json, os
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


#####################################################################################################################
# READ IN
#####################################################################################################################
# 1 Read in the data preprocessed by wikiextractor by Prof. Attardi 
# http://medialab.di.unipi.it/wiki/Wikipedia_Extractor
def read_in_wikiextractor_output():
	path = "/home/sscepano/Project7s/Twitter/wiki_test_learn/OUTPUT/wikiALL_titles_url_out_no_templates"
	#path = "/home/sscepano/Project7s/Twitter/wiki_test_learn/INPUT/testINPUT"

	# count how many articles (i.e., lines) are read
	cnt_articles = 0
	article_ids = defaultdict(int)
	article_urls = defaultdict(int)
	train_set_dict = defaultdict(int)
	# I adapted wikiextractor to output one article per single line
	# here we read in such output
	# read in all the files from all the folders (Attardi divides output in many files in many folders)
	for dirin_name in os.listdir(path):
		print dirin_name
		dirin_name_full = os.path.join(path, dirin_name)
		for fin_name in os.listdir(dirin_name_full):
			#print fin_name
			fin_name_full = os.path.join(dirin_name_full, fin_name)
			print fin_name_full
			fin = open(fin_name_full,"r")
			# for each article i.e., line
			for line in fin:
				# disambiguation page check
				suffix = "may refer to:"
				if not (line[:-1].endswith(suffix)):
					line = line[:-1].split(" ")
					# article minimum length check
					if (len(line) >= 200):
						# take first number in the line, that is the id
						aid = line[0]
						# second element is the url
						aurl = line[1]
						# join back (not so efficient) the rest of the line
						aline = " ".join(line[2:])
						# the dictionaries save what they need to save
						article_ids[cnt_articles] = int(aid)
						article_urls[cnt_articles] = aurl
						# append the line == article text to train set
						train_set_dict[cnt_articles] = aline
						cnt_articles += 1
			fin.close()
	print "INSERTED articles: ", cnt_articles
	return cnt_articles, article_ids, article_urls, train_set_dict


# 2 read in the data preprocessed by Wiki preprocessor given by Gabrilovich
# 
def read_inlinks(f = "20051105_pages_articles.stat.inlinks"):
	articles_stat_links = defaultdict(int)
	for line in open(f, "r"):
		page, links = line.split()
		page = int(page)
		links = int(links)
		articles_stat_links[page] = links
	return articles_stat_links

def process_keywords(aid, text, title, cnt_articles, article_ids, train_set_dict, outfile):
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
			print cnt_articles, aid, title
		outfile.write(str(aid) + '\t' + str(cnt_articles) + '\t' + str(title) + '\n')
		cnt_articles += 1
	return cnt_articles

def process_page(buf, cnt_articles, article_ids, train_set_dict, outfile):
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
	outlinks = page.attrib['outlinks']
	aid = page.attrib['id']
	if page.attrib['stub'] == "1":
		return cnt_articles
	if int(outlinks) < 5:
		return cnt_articles
	if asl[int(aid)] < 5:
		return cnt_articles
	t = re.search(r'<title>(.*)</title>', buf, re.DOTALL)
	title = t.group(1) if t else None
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
	# finally, if page passed the pruning phase, extract only its text
	m = re.search(r'<text>(.*)</text>', buf, re.DOTALL)
    text = m.group(1) if m else None
    # I choose not to care about headers and other xml markup that might be left
	text = re.sub(r'<(.*)>', '', text, re.DOTALL)
	# for the next step, I need one-line articles, so we replace new lines with spaces
	text = text.replace('\n', ' ')
	# finally process the text for non-stopwords (conceptually that code could have been here, but is cleaner this way)
	cnt_articles = process_keywords(aid, text, title, cnt_articles, article_ids, train_set_dict, outfile)
	return cnt_articles

def process_hgw_xml(f_in = "20051105_pages_articles.hgw.xml",f_articles_out = "v4_AID_hgw_titles.tsv"):
	# count how many articles (i.e., lines) are read
	outfile = open(f_articles_out,'w')
	outfile.write('_id' + '\t' + 'our_id' + '\t' + 'title' + '\n')
	cnt_articles = 0
	cnt_all_articles = 0
	article_ids = defaultdict(int)
	train_set_dict = defaultdict(int) 
	inputbuffer = ''
	with open(f_in,'r') as input_file:
		# the code loops through the input, forms pages when they are found, and sends them to process_page for further steps
	    append = False
	    for line in input_file:	
			if '<page id=' in line:
				inputbuffer = line
				append = True
	  		elif '</page>' in line:
				inputbuffer += line
				cnt_all_articles += 1
	 			append = False
				try:
					cnt_articles = process_page(inputbuffer, cnt_articles, article_ids, train_set_dict, outfile)
				except TypeError as e:
					print e
				inputbuffer = None
				del inputbuffer #probably redundant...
			elif append:
				inputbuffer += line
	outfile.close()
	print "INSERTED articles: ", cnt_articles, "ALL READ articles: ", cnt_all_articles
	return cnt_articles, article_ids, train_set_dict


#cnt_articles, article_ids, train_set_dict = process_hgw_xml()
#####################################################################################################################

#####################################################################################################################
# TF-IDF
#####################################################################################################################
# Code that calculates TF-IDF is adapted from the blogpost by Christian S. Perone
# http://blog.christianperone.com/?p=1589
# The code uses scikits.learn Python module for machine learning http://scikit-learn.sourceforge.net/stable/

# 1 Normalize TF-IDF (Gabrilovich)
def tfidf_normalize(cnt_articles, article_ids, article_urls = None, train_set_dict):
	# use the whole (Wiki) set as both the train and test set
	train_set = train_set_dict.values()
	test_set = train_set
	# instantiate vectorizer with English language, using stopwords and set min_df, max_df parameters and the tokenizer
	vectorizer = CountVectorizer(stop_words='english', min_df=3, max_df=0.7, token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b')
	# by appling the vectorizer instance to the train set
	# it will create a vocabulary from all the words that appear in at least min_df and in no more than max_df
	# documents in the train_set
	vectorizer.fit_transform(train_set)
	# vectorizer transform will apply the vocabulary from the train set to the test set. In my case,
	# they are the same set: whole Wikipedia.
	# this means that each article will get representation based on the words from the vocabulary and
	# their TF-IDF values in the Scipy sparse output matricx
	freq_term_matrix = vectorizer.transform(test_set)
	# this is a log transformation as applied in (Gabrilovich, 2009), i.e., that is
	# how he defines TF values. In case of TF = 0, this shall not affect such value
	# freq_term_matrix.data = 1 + np.log( freq_term_matrix.data )
	# instantiate tfidf trnasformer
	tfidf = TfidfTransformer(norm="l2", sublinear_tf=True)
	# tfidf uses the freq_term_matrix to calculate IDF values for each word (element of the vocabulary)
	tfidf.fit(freq_term_matrix)
	# finally, tfidf will calculate TFIDF values with transform()
	tf_idf_matrix = tfidf.transform(freq_term_matrix)
	# now we put our matrix to CSC format (as it helps with accessing columns for inversing the vectors to
	# words' concept vectors)
	CSC_matrix = tf_idf_matrix.tocsc() 
	# we need vocabulary_ to be accessible by the index of the word so we inverse the keys and values of the
	#dictionary and put them to new dictionary word_index
	word_index = dict((v, k) for k, v in vectorizer.vocabulary_.iteritems())
	M, N = CSC_matrix.shape
	print "Articles: ", M
	print "Words: ", N
	return M, N, CSC_matrix, word_index

# 2 Raw TF-IDF values (Hieu)
def tfidf_raw(cnt_articles, article_ids, article_urls = None, train_set_dict):
	# use the whole (Wiki) set as both the train and test set
	train_set = train_set_dict.values()
	test_set = train_set
	# instantiate vectorizer with English language, using stopwords and set min_df, max_df parameters and the tokenizer
	vectorizer = CountVectorizer(stop_words='english', min_df=3, max_df=0.7, token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b')
	# by appling the vectorizer instance to the train set
	# it will create a vocabulary from all the words that appear in at least min_df and in no more than max_df
	# documents in the train_set
	vectorizer.fit_transform(train_set)
	# vectorizer transform will apply the vocabulary from the train set to the test set. In my case,
	# they are the same set: whole Wikipedia.
	# this means that each article will get representation based on the words from the vocabulary and
	# their TF-IDF values in the Scipy sparse output matricx
	freq_term_matrix = vectorizer.transform(test_set)
	# this is a log transformation as applied in (Gabrilovich, 2009), i.e., that is
	# how he defines TF values. In case of TF = 0, this shall not affect such value
	# freq_term_matrix.data = 1 + np.log( freq_term_matrix.data )
	# instantiate tfidf trnasformer
	tfidf = TfidfTransformer(norm=None, sublinear_tf=False)
	# tfidf uses the freq_term_matrix to calculate IDF values for each word (element of the vocabulary)
	tfidf.fit(freq_term_matrix)
	# finally, tfidf will calculate TFIDF values with transform()
	tf_idf_matrix = tfidf.transform(freq_term_matrix)
	# now we put our matrix to CSC format (as it helps with accessing columns for inversing the vectors to
	# words' concept vectors)
	CSC_matrix = tf_idf_matrix.tocsc() 
	# we need vocabulary_ to be accessible by the index of the word so we inverse the keys and values of the
	#dictionary and put them to new dictionary word_index
	word_index = dict((v, k) for k, v in vectorizer.vocabulary_.iteritems())
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
def save_CV_with_sliding_window_pruning(m, fn, window=100, drop_pct=5):
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
			highest_scoring_concept_pct = highest_scoring_concept * drop_pct/100.0
			remembered_tfidf = highest_scoring_concept
			k = 0
			for (idx, value) in sorted(zip(col.indices, col.data), key = lambda t: t[1], reverse=True):
				k += 1
				tfidf_dict = {}
				tfidf_dict[str(idx)] = (str(value),str(article_ids[idx]))
				CV_dict['CV'].append(tfidf_dict)
				if k % window == 0:
					if remembered_tfidf - value > highest_scoring_concept_pct:
						chck_pruning = += len(col.data) - k
						break
					else:
						remembered_tfidf = value
			f.write(unicode(json.dumps(CV_dict, ensure_ascii=False)) + '\n')
			if CNT % 10000 == 0:
				print CNT, the_word, highest_scoring_concept, value
	print "1 Pruned terms total: ", chck_pruning

# 2 save with threshold pruning (Hieu)
def save_CV_with_threshold_pruning(m, fn, threshold=12):
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
	print "2 Pruned terms total: ", chck_pruning
#####################################################################################################################


cnt_articles, article_ids, article_urls, train_set_dict = read_in_wikiextractor_output()
M, N, CSC_matrix, word_index = tfidf(cnt_articles, article_ids, article_urls, train_set_dict)
# file to save the matrix
filename2 = "tf_idf_ALL137s.json"
# save_CV(testM, filename2)
save_CV_with_pruning(CSC_matrix, filename2)

fn_test = "test_CV1311.json"
test_before_save_CV_with_pruning(CSC_matrix, fn_test)

test = [7,4,3,2,1,7,7]


'''
author: sanja7s

V13: default tokenizer will now be instead token_pattern=u'(?u)\b\w\w+\b'
token_pattern=u'(?u)\b[a-zA-Z][a-zA-Z]+\b' NOno this is r'\b[a-zA-Z][a-zA-Z]+\b'
*Note \w == [a-zA-Z0-9_] and I think I don't need either 0-9 either _
Let's try.
token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'


V12: Hieus method is not normalized, no log transformation, we just use raw TF values. I will now once again try that, but using
parameters min_df = 3 and max_df = 0.7
reasoning: if we want to compare words based on similar concepts, then we probably also want those to have similar tfidf scales.
ok, article lengths also represent to us their importance in "general" in some way. perhaps :)

I am checking for a really small value >= 2 so that we get almost all concepts and then i can try thresholding as Hieu

this gave output og 19G and many lines biggern than 16MB. That pointed me out to many numbers trated as token == words in my
case and i want to remove them. After short reading on regular epxressions, I think I will just change the default tokenizer
in the next version.

V11: the check while saving would be 
if value > 0:
which means basically to save everything. but I am afraid that would mean that I will have some json element
larger than 16MB and then I cannot save it.

so instead, I will implement some check. I will implement the one as descirbed in (Gavrilovich, 2009)
 

V10: like V9 just applying the log tranform 
tfidf = TfidfTransformer(norm=None, sublinear_tf=True)

V9: going back to max_df = 0.7 as the previously tried 0.1 causes loss of important words such as time and life
however, unlike V7, we now keep min_df = 3, so now I can measure a case that makes a bit more sense according to me
and that was mentioned by Gabrilovich (he does not introduce max_df, so in the end, if needed, I can try without it, too)
Articles #:  1829625
Words #:  1294606
Also, I did not apply the log tranform here

V8: going back to max_df = 0.1. Also, we thought that my min_df is maybe too large.
Because this means that I ask that the term appears 8 times in different documents;
Gabrilovich used here min_df = 3, so I will try that now, too.
M, N = CSC_matrix.shape
Articles #:  1829625
Words #:  1294321
I also changed here the value where I cut off while saving -- to be 8

V7: we added normalization of freq_term_matrix with Antti as below:
freq_term_matrix.data = 1 + np.log( freq_term_matrix.data )
so that we follow approach by Gabrilovich
also we improved some parts of my code with zip and also saved for the first time actually CSR matrix


V6: first implmentation with TF-IDF as non-normalized. In this way, I can test Hieu's
approach, since his results are better than the original; and he used no filtering of
articles. I have put here min_df = 8 and max_df = 0.7. The second paramether perhaps 
makes sense in a smaller corpus, but in this one I was probably better of using 0.1
as suggested by Hieu.

CHECK: if articles with dates are still present here

DONE: pruning disambiguation pages with "may refer to:"
      pruning categories done in previous step, wikiextractor by Attardi
      pruning too short articles (< 200 words)
      pruning rare words (< 3 articles) (min_df)
      pruning too frequent (= corpus specific stopwords) present in >10% articles (max_df)

This code calculates a knowledge base for semantic relatendess (SR) from Wikipedia data
according to Explicit Semantic Analysis (ESA) algorithm [Gabrilovich et al. 2007].

The input data is beforehand preprocessed from an English Wikipedia dump
2015-03-07 (enwiki-20150304-pages-articles.xml.bz2 10.9 GB) 
using the wikiextractor code from https://github.com/attardi/wikiextractor by Giuseppe Attardi.

TF-IDF is calculated for the words in all the articles. We use sklearn TfidfTransformer
that builds a vocabulary omitting English stop_words, with minimum document frequency for a word 3,
and maximum percent of articles in which the word appears 10%.
TfidfTransformer gives us a sparse scipy matrix that has words as columns, articles as rows,
and corresponding TF-IDF values as elements.
The sparce scipy matrix is well suited for our case since each article contains a small % of the words, 
so the resulting TF-IDF matrix is quite sparse.

From such a TF-IDF matrix, we obtain a concept vector (CV) for each word, containing articles and
corresponding TF-IDF values. We threshold the TF-IDF values with 0.12 based on (Hieu et al. 2013).
The concept vectors are obtained by slicing columns of scipy matrix.

The final output of this code is a text file formated as a set of single line json objects
that is suitable to be imported to MongoDB where our knwoledge base will be stored.


References
----------
Gabrilovich, Evgeniy, and Shaul Markovitch. "Computing Semantic Relatedness Using Wikipedia-based Explicit Semantic Analysis." IJCAI. Vol. 7. 2007.
Hieu, Nguyen Trung, Mario Di Francesco, and Antti Ylä-Jääski. "Extracting knowledge from wikipedia articles through distributed semantic analysis." Proceedings of the 13th International Conference on Knowledge Management and Knowledge Technologies. ACM, 2013.
Gabrilovich, Evgeniy, and Shaul Markovitch. "Wikipedia-based semantic interpretation for natural language processing." Journal of Artificial Intelligence Research (2009): 443-498.
'''
import io, json, os
import numpy as np
import scipy
# dict for zipping article ids and urls with the counter ids from this code  
from collections import defaultdict
# CountVectorizer counts the number of times a term occurs in the document
from sklearn.feature_extraction.text import CountVectorizer
# TfidfTransformer will use output from CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import itertools
import collections

path = "/home/sscepano/Project7s/Twitter/wiki_test_learn/OUTPUT/wikiALL_titles_url_out_no_templates"
#path = "/home/sscepano/Project7s/Twitter/wiki_test_learn/INPUT/testINPUT"

# count how many articles (i.e., lines) are read
cnt_articles = 0
article_ids = defaultdict(int)
article_urls = defaultdict(int)
train_set_dict = defaultdict(int) 

# read all the files in from all the folders
# remove "disambiguation" pages with check for "may refer to:"
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
			# disambiguation check
			suffix = "may refer to:"
			if not (line[:-1].endswith(suffix)):
				line = line[:-1].split(" ")
				# article minimum length check
				if (len(line) >= 200):
					# take first number in the line, that is the id
					aid = line[0]
					# second element is the url
					aurl = line[1]
					# join back (not so efficient) the rest of the line TODO CHCK
					aline = " ".join(line[2:])
					# the dictionaries save what they need to save
					article_ids[cnt_articles] = int(aid)
					article_urls[cnt_articles] = aurl
					# append the line == article text to train set
					train_set_dict[cnt_articles] = aline
					cnt_articles += 1
		fin.close()

print "INSERTED articles #: ", cnt_articles

train_set = train_set_dict.values()
# use the whole (Wiki) set as the train set
test_set = train_set

# instantiate vectorizer with English language, using stopwords and
# min_df = 3 and max_df = 70%
#vectorizer = CountVectorizer(stop_words='english', min_df=3, max_df=0.7, token_pattern=u'(?u)\b[a-zA-Z][a-zA-Z]+\b')
#vectorizer = CountVectorizer(stop_words='english', min_df=3, max_df=0.7, tokenizer = word_tokenize, token_pattern=u'(?u)\b[a-zA-Z][a-zA-Z]+\b')
#vectorizer = CountVectorizer(stop_words='english', min_df=3, max_df=0.7, analyzer=partial(regexp_tokenize, pattern=u'(?u)\b[a-zA-Z][a-zA-Z]+\b'))
vectorizer = CountVectorizer(stop_words='english', min_df=3, max_df=0.7, token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b')

# we apply the vectorizer to the train set. 
# it will count all the words that appear at least 3 times in one article
# and in no more than 70% as part of my vocabulary_
vectorizer.fit_transform(train_set)

# vectorizer transform will apply the vocabulary from the train set to the
# test set. In my case, they are the same set: whole Wikipedia.
freq_term_matrix = vectorizer.transform(test_set)

# this is a log transformation as applied in (Gabrilovich, 2009), i.e., that is
# how he defines TF values. In case of TF = 0, this shall not affect such value
#freq_term_matrix.data = 1 + np.log( freq_term_matrix.data )

# let me filter this one. I want to remove all articles == concepts
# that are left with less than 100 important words == terms idenitifed here
# that will be a bit more strict than in the paper (Gabrilovtch & Markovitch, 2009)
# but since Wikipedia is by now bigger, that should work fine

M1, N1 = freq_term_matrix.shape
# freq_term_matrix to calculate tfidf values
tfidf = TfidfTransformer(norm=None, sublinear_tf=False)
# tfidf uses the freq_term_matrix to calculate IDF
tfidf.fit(freq_term_matrix)
# finally, tfidf will calculate TFIDF values with transform()
tf_idf_matrix = tfidf.transform(freq_term_matrix)

'''
# function to save to file the raw CSR 
# just in case somthing breaks with my pritning of the desired txt output from
# the CSR/CSC matrix, I can use this function to save it as it is in npy format
def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )
# just in case save the matrix
filename = "tf_idf_CSR_12"
save_sparse_csr(filename,tf_idf_matrix)
'''

# now we put our matrix to CSC format (as it helps with accessing columns
# for inversing the vectors to words concept vectors)
CSC_matrix = tf_idf_matrix.tocsc() 

# we will need vocabulary_ to be accessible by the index of the word
# so we inverse the keys and values of the dictionary
word_index = dict((v, k) for k, v in vectorizer.vocabulary_.iteritems())

M, N = CSC_matrix.shape
print "Articles #: ", M
print "Words #: ", N

# save desired output for the next step (input to MongoDB)
# CV = word, article_id1:tfidf1 ... article_idN:tfidfN *FORMAT 4 JSON*
# we go through columns and print the word and the indices (= article_id)
# and the column data (= tfidf)
# indices tell the right article_id for those values
def save_CV_with_pruning(m, fn):
	M, N = m.shape
	with io.open(fn, 'w', encoding='utf-8') as f:
		for j in range(N):
			CV_dict = {}
			the_word = word_index[j]
			CV_dict['_id'] = the_word
			CV_dict['CV'] = []
			col = m.getcol(j)
			for (idx, value) in sorted(zip(col.indices, col.data), key = lambda t: t[1], reverse=True):
				if value >= 6:
					tfidf_dict = {}
					tfidf_dict[str(idx)] = (str(value),str(article_ids[idx]))
					CV_dict['CV'].append(tfidf_dict)
				else:
					continue
			f.write(unicode(json.dumps(CV_dict, ensure_ascii=False)) + '\n')

def test_before_save_CV_with_pruning(m, fn):
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
			highest_scoring_concept = max(col.data)
			for (idx, value) in sorted(zip(col.indices, col.data), key = lambda t: t[1], reverse=True):
				if value >= 8:
					tfidf_dict = {}
					tfidf_dict[str(idx)] = (str(value),str(article_ids[idx]))
					CV_dict['CV'].append(tfidf_dict)
			#if len(CV_dict['CV']) > 500:
				#print len(CV_dict['CV']), the_word, highest_scoring_concept, value
			#f.write(unicode(json.dumps(CV_dict, ensure_ascii=False)) + '\n')
			if len(CV_dict['CV']) > 500:
				print len(CV_dict['CV']), the_word, highest_scoring_concept, value
			if CNT == 6000:
				return

# file to save the matrix
filename2 = "tf_idf_ALL137s.json"
# save_CV(testM, filename2)
save_CV_with_pruning(CSC_matrix, filename2)

fn_test = "test_CV1311.json"
test_before_save_CV_with_pruning(CSC_matrix, fn_test)

test = [7,4,3,2,1,7,7]


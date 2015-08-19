import pymongo
from pymongo import MongoClient
from collections import defaultdict, OrderedDict
import math, numpy
import copy
from bson.objectid import ObjectId
from pydoc import help
from scipy.stats.stats import pearsonr, spearmanr

# function takes any two words, looks up their CVs in the collection
# exctracts the CVs and invokes standard vector cosine similarity
def SR_2words(w1, w2):
	CV, AID7s = connection_params()
	cv1 = CV.find_one({"_id": w1})
	cv2 = CV.find_one({"_id": w2})
	if cv1 == None or cv2 == None:
		return -1
	v1 = extract_CV(cv1)
	v2 = extract_CV(cv2)
	return cosine_2vectors_full(v1, v2)

# given two vectors as dictionaries with *ANY* sets of keys
# return their cosine similarity *vectors may be of ANY dim*
# cosine sim (v1,v2) = v1.v2 / ||v1|| ||v2||
def cosine_2vectors_full(v1, v2):
	# numerator for the cosine formula 
	SR_num = 0.0
	# two denominator terms in the formula
	v1_sq_sum = 0.0
	v2_sq_sum = 0.0
	keys_1 = set(v1.keys())
	keys_2 = set(v2.keys())
	# separate the different keys and common keys
	different2_keys = keys_2 - keys_1
	different1_keys = keys_1 - keys_2
	common_keys = keys_1 & keys_2
	#print common_keys
	# for common keys, we calculate formula as is
	# SR = v1.v2 / ||v1|| ||v2||
	for term in common_keys:
		v1_fq = v1[term]
		v2_fq = v2[term]
		SR_num += v1_fq * v2_fq
		v1_sq_sum += v1_fq * v1_fq
		v2_sq_sum += v2_fq * v2_fq
	# for different keys, we just take resepective non-zero 
	# dict terms for calculating denominator (nominator is zero)
	for term in different1_keys:
		v1_fq = v1[term]
		v1_sq_sum += v1_fq * v1_fq
	for term in different2_keys:
		v2_fq = v2[term]
		v2_sq_sum += v2_fq * v2_fq
	# sum all in denominator and sqrt in the end
	SR_den = math.sqrt(v1_sq_sum*v2_sq_sum)
	try:
		SR = SR_num/SR_den
	except ZeroDivisionError:
		SR = 0
	return SR


# extract the CV vector in the form for calculation with true ids for articles
def extract_CV(cv):
	v = cv['CV']
	if v == None:
		return
	vec = defaultdict(int)
	for el in v:
		for key, value in el.iteritems():
			vec[value[1]] = float(value[0])
	return OrderedDict(sorted(vec.items(), key=lambda x: x[1], reverse= False))



def print_topN_concepts(w, topN):
	CV, AID7s = connection_params()
	print CV
	print AID7s
	cv1 = CV.find_one({"_id": w})
	v1 = extract_CV(cv1)
	print w, " has ", len(v1.items()), " concepts CV "
	print "Top concepts are: "
	for i in range (topN):
		term = v1.popitem()
		#print term[0], term[1]
		#print type(term[0])
		article_name = AID7s.find_one({"_id": long(str(term[0]))})
		print term[1], article_name


def read_in_human_judgement(file_name = "SR_human_similarity/Mturk.csv"):
	if '.csv' in file_name:
		sep = ','
	else:
		sep = '\t'
	f = open(file_name, "r")
	human_SR = defaultdict(int)
	#header = f.readline()
	for line in f:
		#w1, w2, SR = line.split('\t')
		els = line.split(sep)
		w1, w2, SR = els[0], els[1], els[2]
		human_SR[(w1,w2)] = float(SR)
	return human_SR

def evaluate_Wiki_DB_against_human():
	human_SR  = read_in_human_judgement()
	human_SR_lst = []
	Wiki_SR_lst = []
	Wiki_Nterms_SR_lst = []
	Wiki_threshold_SR_lst = []
	for el in human_SR.iteritems():
		w1 = el[0][0]
		w2 = el[0][1]
		h_SR = el[1]
		Wiki_SR = SR_2words(w1.lower(), w2.lower())
		if Wiki_SR <> -1:
			human_SR_lst.append(h_SR)
			Wiki_SR_lst.append(Wiki_SR)
		print w1, w2, h_SR, Wiki_SR
	print human_SR_lst
	print Wiki_SR_lst
	print "Full Wiki Pearson ", pearsonr(human_SR_lst, Wiki_SR_lst)
	print
	print "Full Wiki Spearman ", spearmanr(human_SR_lst, Wiki_SR_lst)


#############################################################################################
# take the right COLLECTION = TF-IDF based concept vectors (CV) -- output from tfidf.py
#############################################################################################
def connection_params(client = MongoClient(), dbs="test", CV_collection="CV_Gab2005_true_selected_by_Gab_pruned_loglog", aid_collection="v3_AID"):
	# connect to Mongo db test
	client = MongoClient()
	db = client[dbs]
	CV = db[CV_collection]
	AID7s = db[aid_collection]
	return CV, AID7s
#############################################################################################


# TEST
print_topN_concepts("weather", 7)

SR_2words("weather", "forecast")

evaluate_Wiki_DB_against_human()

# for general error handling
import io, sys
import traceback
import logging
#
import read_in_wiki as r
import tfidf as t
import save_CV as s
#####################################################################################################################
# Execute track 1 Gabrilovich
#####################################################################################################################
F_IN = "/media/sscepano/Data/Wiki2015/wikiextractor-master/text"
F_IN_TEST = "/media/sscepano/Data/Wiki2015/test_stuff/text.100K"
F_OUT_AID = "/media/sscepano/Data/Wiki2015/articles_selected"
F_OUT_CV = "/media/sscepano/Data/Wiki2015/CV.json"

def ESA():
	global F_IN
	global F_OUT_CV
	global F_OUT_AID
	_log = logging.getLogger(__name__)
	logging.basicConfig(level=logging.INFO, format='%(name)s: %(levelname)-8s %(message)s')

	articles_with_aid = r.read_in_wikiextractor_output(F_IN, F_OUT_AID)
	#print r.articles_selected(777)
	_log.info("*** Data loaded. ***")

	tf_idf_matrix, word_index, long_articles = t.tfidf_normalize(articles_with_aid)

	s.save_CV_all(tf_idf_matrix, word_index, long_articles, F_OUT_CV)


	while True:
		# enter 1 if you want tfidf processing to be performed, else saving the data will be only done
		c = raw_input("Press enter to start a process cycle:\n")
		try:
			reload(t)
		except NameError:
			_log.error("Could not reload the module.")
		try:
			if int(c) == 1:
				tf_idf_matrix, word_index, long_articles = t.tfidf_normalize(articles_with_aid)
				_log.info("*** Tfidf calculation successfully done. ***")
		except:
			e = sys.exc_info()[0]
			_log.error("Caught exception from the process\n%s\n%s" % (e, traceback.format_exc()))	
		try:
			try:
				reload (s)
			except NameError:
				_log.error("Could not reload the module.")
			if int(c) == 2:
				s.save_CV_all(tf_idf_matrix, word_index, long_articles, F_OUT_CV)
				_log.info("*** Data successfully saved. ***")
		except:
			e = sys.exc_info()[0]
			_log.error("Caught exception from the process\n%s\n%s" % (e, traceback.format_exc()))
		
		_log.info("Cycle ready.\n")


#####################################################################################################################


ESA()
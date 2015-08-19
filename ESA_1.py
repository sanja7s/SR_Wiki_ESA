# for general error handling
import io, sys
import traceback
import logging
#
import tfidf as t

#####################################################################################################################
# Execute track 1 Gabrilovich
#####################################################################################################################


def ESA_1(fn="output/CV_Gab2005_STOP_CATS.json"):
	_log = logging.getLogger(__name__)
	logging.basicConfig(level=logging.INFO, format='%(name)s: %(levelname)-8s %(message)s')

	cnt_articles, article_ids, train_set_dict = t.process_hgw_xml()
	_log.info("*** Data loaded. ***")

	while True:
		# enter 1 if you want tfidf processing to be performed, else saving the data will be only done
		s = raw_input("Press enter to start a process cycle:\n")
		try:
			reload(t)
		except NameError:
			_log.error("Could not reload the module.")
		try:
			if int(s) == 1:
				M, N, CSC_matrix, word_index = t.tfidf_normalize(cnt_articles, article_ids, train_set_dict)
				_log.info("*** Tfidf calculation successfully done. ***")
		except:
			e = sys.exc_info()[0]
			_log.error("Caught exception from the process\n%s\n%s" % (e, traceback.format_exc()))	
		try:
			t.save_CV_all(CSC_matrix, fn, word_index, article_ids)
			_log.info("*** Data successfully saved. ***")
		except:
			e = sys.exc_info()[0]
			_log.error("Caught exception from the process\n%s\n%s" % (e, traceback.format_exc()))
		
		_log.info("Cycle ready.\n")

#####################################################################################################################


ESA_1()
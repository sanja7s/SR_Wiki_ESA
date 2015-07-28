# for general error handling
import io, sys
import traceback
import logging
#

import tfidf as t

#####################################################################################################################
# Execute track 1 Gabrilovich
#####################################################################################################################


def ESA_1(fn="output/CV_Gabrilovich_data2005_sample_v2.json"):
	_log = logging.getLogger(__name__)
	logging.basicConfig(level=logging.INFO, format='%(name)s: %(levelname)-8s %(message)s')

	cnt_articles, article_ids, train_set_dict = t.process_hgw_xml(f_in = "Gabrliovich_preprocessed/sample.hgw.xml")
	_log.info("*** Data loaded. ***")

	tfidf_done = False
	while True:
		raw_input("Press enter to start a process cycle:\n")
		try:
			reload(t)
		except NameError:
			_log.error("Could not reload the module.")
		try:
			if not tfidf_done:
				M, N, CSC_matrix, word_index = t.tfidf_normalize(cnt_articles, article_ids, train_set_dict)
				tfidf_done = True
				_log.info("*** Tfidf calculation successfully done. ***")
		except:
			e = sys.exc_info()[0]
			_log.error("Caught exception from the process\n%s\n%s" % (e, traceback.format_exc()))
		
		try:
			t.save_CV_with_sliding_window_pruning(CSC_matrix, fn, word_index, article_ids)
			_log.info("*** Data successfully saved. ***")
		except:
			e = sys.exc_info()[0]
			_log.error("Caught exception from the process\n%s\n%s" % (e, traceback.format_exc()))
		
		_log.info("Cycle ready.\n")

#####################################################################################################################


ESA_1()
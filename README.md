# SR_Wiki_ESA
Implementation of ESA in Python using MongoDB and sql table from Wiki dumps for categories.

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

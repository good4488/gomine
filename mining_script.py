# -*- coding:utf-8 -*-
import os
import time
import numpy as np
import random as rn

from Preprocessing import *
from rake import *
from Textmining import *

start_time = time.time()

# =================================================
#   Variables
# =================================================
Num_files = 3000
Random_option = False
N_gram = (1,1)
MAX_DF, MIN_DF = 1.0, 1
ALL_WORD = True
direc_suffix = '/160911_test2/'
COREMOF, GSCHOLAR, PUBMED = False, False, True

# =================================================
#   Directory Setting
# =================================================
main_path = os.getcwd()
html_path = main_path+'/articles_html/'
data_path = main_path+'/data'
raw_path = data_path+direc_suffix+'raw_text/'
full_path = data_path+direc_suffix+'full_text/'
out_path = data_path+direc_suffix+'Output/'
token_path = data_path+direc_suffix+'tokens/'
TFIDF_path = data_path+direc_suffix+'TFIDF/'
ONLYTF_path = data_path+direc_suffix+'ONLYTF/'
RAKE_path = data_path+direc_suffix+'RAKE_keywords/'


WRITE_PATH_LIST = (raw_path, full_path, out_path, token_path, TFIDF_path, ONLYTF_path, RAKE_path)

control_all_directory(data_path, direc_suffix, WRITE_PATH_LIST)
# =================================================
#   Make Text file?
# =================================================
MAKE_FULL_TEXT = True
MAKE_RAW_TEXT = True
BUILD_CORPUS = True
REUSE_CORPUS = True
REUSE_PATH = data_path+'/160911_test/tokens/'

TFIDF, ONLYTF = True, False
RAKE = False

# =================================================
#   Make Output files?
# =================================================
OUT_TFIDF = True
OUT_ONLYTF = True
OUT_COS_SIM = True

# =================================================
#   Use Stopword and setting options
# =================================================
stop_words = get_stopword_set()

# =================================================
BASIC_PATH = (main_path, html_path, raw_path, full_path, token_path, out_path, TFIDF_path, ONLYTF_path, RAKE_path)
CORPUS_OPTION = (MAKE_FULL_TEXT, MAKE_RAW_TEXT, BUILD_CORPUS, REUSE_CORPUS, RAKE, COREMOF, GSCHOLAR, PUBMED)
OUTPUT_OPTION = (OUT_TFIDF, OUT_ONLYTF, OUT_COS_SIM)
TFIDF_OPTION = (TFIDF, OUT_TFIDF, MAX_DF, MIN_DF, N_gram, ALL_WORD)
ONLYTF_OPTION = (TFIDF, OUT_ONLYTF, MAX_DF, MIN_DF, N_gram, ALL_WORD)
# =================================================
#   main script
# =================================================

# get file_list from text file
file_list = get_file_list(main_path+'/list_time.txt', Num_files, Random_option)


# make corpus from HTML paper to text
corpus_maker = Make_custom_corpus(file_list, CORPUS_OPTION, BASIC_PATH, stop_words)
total_corpus = corpus_maker.build_corpus(REUSE_PATH, Num_files)
print('Making Corpus End......')


# #####Machine Learning class


# TFIDF
TFIDF_model = TFIDF_Vectorizer_scikit(total_corpus, TFIDF_OPTION, BASIC_PATH)
TFIDF_model.run()

#TFIDF_keyword = TFIDF_model.keyword_analyzer()
#occur_matrix = TFIDF_model.build_co_occur_matrix()
#std_list = TFIDF_model.get_std_list()
TFIDF_total_word = TFIDF_model.get_unique_word()
#Idea_list = TFIDF_model.get_test_idea()


df_list = TFIDF_model.get_df_list()
idf_list = TFIDF_model.get_idf_list()
target_keyword = TFIDF_model.list_for_keywordset(20)

'''
import setlink
KeywordSetModel = setlink.KeywordSet(TFIDF_total_word,target_keyword, 1, BASIC_PATH)
KeywordSetModel.run()

print(' by file index ')
interset = KeywordSetModel.get_intersection_byfileindex([955,1661])
words = [TFIDF_total_word[word] for word in interset]
print(words)

print(' by wordlist' )
inter_index = KeywordSetModel.get_interfile_byword(['fullerene','hydrogen'])
print(inter_index)
inter_index = KeywordSetModel.get_interfile_byword(['conductivity','tcnq'])
print(inter_index)
inter_index = KeywordSetModel.get_interfile_byword(['screening','conductivity'])
print(inter_index)
inter_index = KeywordSetModel.get_interfile_byword(['screening','conductivity','tcnq'])
print(inter_index)
#inter_index = KeywordSetModel.get_interfile_byword(['fullerene','hydrogen'])
#print(inter_index)


print(' by regex ' )
reg_index = KeywordSetModel.get_interfile_byregex(['tcnq','conduct'])
print(reg_index)
print(' by regex ' )
reg_index = KeywordSetModel.get_interfile_byregex(['screen','conduct'])
print(reg_index)
print(' by regex ' )
reg_index = KeywordSetModel.get_interfile_byregex(['tcnq','conduct','screen'])
print(reg_index)

#norm_tf_matrix = TFIDF_model.get_norm_tf()
#print(df_list[0], df_list[1])


# Word vs. df
df_vs_word_dic = document_frequency_versus_word(out_path, df_list, TFIDF_total_word)
index_versus_word(out_path)

# ONLYTF
# ONLYTF_model = ONLYTF_Vectorizer_scikit(total_corpus, ONLYTF_OPTION, BASIC_PATH)
# ONLYTF_model.run()


# F-measure
# F_measure_model = Get_Recall_Precision(maunal_keyword, mine_keyword)


# Keyword_analyzer
# ================
# Manual assigned keywords



# ================
# 1. TF * a(IDF)


# ================
'''







# End
end_time = time.time()-start_time
print('Run_time =  %f'%end_time)
print('JOB DONE')

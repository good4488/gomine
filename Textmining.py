# -*- coding:utf-8 -*-
import os, string, re, nltk, codecs, scipy
import random as rn
import numpy as np
import os.path

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
import matplotlib
import matplotlib.pyplot as plt
import itertools

from Preprocessing import *
from rake import *
from pubmed import *
import setlink

remove_tag_list = ["script", "style","table","ui","li","a","ol","noscript",'option']
lemmatizer = WordNetLemmatizer()

def checkdir_and_makedir(path):
    if os.path.isdir(path):
        print('%s   path    exist!'%path)
        #raise NotImplementedError
    else:
        print('make path:   %s'%path)
        os.mkdir(path)


def check_file_and_read(path):
    if os.path.isfile(path):
        print('READ %s'%path)
        f = open(path, 'r')
        return f.read()
    else:
        print('%s   file does not exist!'%path)
        raise NotImplementedError


def get_stopword_set():
    vectorizer_for_stop = TfidfVectorizer(stop_words='english')
    stop_words = set()
    with open('stopwords_custom.in','r',encoding = 'UTF-8') as f:
        stop_words = stop_words|set(f.read().split())
    with open('SmartStoplist.txt','r',encoding='UTF-8') as f:
        stop_words = stop_words|set(f.read().split())
    stop_words = stop_words|set(stopwords.words('english'))|vectorizer_for_stop.get_stop_words()
    return stop_words


def control_all_directory(data_path, suffix, write_path_list):

    local_path = data_path+suffix
    checkdir_and_makedir(local_path)

    for p in write_path_list:
        checkdir_and_makedir(p)

    print('Directory setting OK...\n')


def get_file_list(path,length,option=0):
    f = open(path,'r')
    files = f.readlines()
    f.close()
    if length > len(files):
        length = len(files)

    print('In list_time, %d file inputs!'%length)
    file_list = []
    for name in files:
        file_list.append(name[:-1])

    if option:
        file_list = file_list[:length]
        rn.shuffle(file_list)
        return file_list
    else:
        file_list = file_list[:length]
        return file_list


def document_frequency_versus_word(path, df_array, word_array):
    df_array, word_array = (list(t) for t in zip(*sorted(zip(df_array,word_array))))
    f = open(path+'df_vs_word.txt','w',encoding='UTF-8')
    df_word_dictionary = dict()

    for i, df in enumerate(df_array):
        if i == 0:
            df_word_dictionary[df] = []
            df_word_dictionary[df].append(word_array[i])
            f.write('%d %s  '%(df,word_array[i]))
        elif df == df_array[i-1]:
            df_word_dictionary[df].append(word_array[i])
            f.write('%s '%(word_array[i]))
        else:
            df_word_dictionary[df] = []
            df_word_dictionary[df].append(word_array[i])
            f.write('\n%d   %s  '%(df,word_array[i]))

    f.close()
    return df_word_dictionary


def prob_based_index(array):
    prob = list(itertools.accumulate(array/sum(array)))
    rand = rn.random()
    for index, p in enumerate(prob):
        if rand <= p:
            return index


def index_versus_word(path):
    index = 1
    outfile = open(path+'index_vs_df.txt','w')
    with open(path+'df_vs_word.txt','r') as f:
        for line in f.readlines():
            tokens = line.split()
            df = tokens[0]
            for i in range(len(tokens[1:])):
                outfile.write(str(index) + " " + df +'\n')
                index = index+1

    outfile.close()


def is_useful_2gram(word):
    # periodic table
    pt_set = ['he', 'li', 'be', 'ne', 'na', 'mg', 'al', 'si', 'cl', 'ar', 'ca', 'sc', 'ti', 'cr', 'mn', 'fe', 'co', 'ni', 'cu', 'zn', 'ga', 'ge'\
    'as', 'se', 'br', 'kr', 'rb', 'sr', 'zr', 'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn', 'sb', 'te', 'xe', 'cs', 'ba', 'hf'\
    'ta', 're', 'os', 'ir', 'pt', 'au', 'hg', 'tl', 'pb', 'bi', 'po', 'at', 'rn', 'fr', 'ra', 'rf', 'db', 'sg', 'bh', 'hs', 'mt', 'ds', 'rg'\
    'cn', 'fl', 'lv', 'la', 'ce', 'pr', 'nd', 'pm', 'sm', 'eu', 'gd', 'tb', 'dy', 'ho', 'er', 'tm', 'yb', 'lu', 'ac', 'th', 'pa', 'np', 'pu'\
    'am', 'bk', 'cf', 'es', 'md', 'no', 'lr']
    high_set = ['ph', 'mo', 'uv', 'ab', 'tg', 'vi', 'di', 'eq', 'gc', 'ch', 'rt', 'fc', 'ta', 'ho', 'nh', 'hf', 'mr', 'cp', 'mc', 'md', 'oh', 'ho', 'nh', 'hn', 'cn', 'nc', 'co'\
    'em', 'er', 'sn', 'sp', 'df', ]
    num_set = ['h2', 'n2', 'o2', 'o3', 'i2']
    if word[-1].isdigit() and word in num_set:
        return True
    if word in pt_set:
        return True
    if word in high_set:
        return True
    else:
        return False


#def is_useful_3gram(word):
    

#def is_useful_alnum(word):

#def suffix_checker(word):



class Make_custom_corpus():
    def __init__(self,file_list, option, basic_path, stop_words):
        self.name_list = file_list
        self.basic_path = basic_path
        self.path_list = [self.basic_path[1]+filename for filename in file_list]
        self.option = option
        self.stop_words = stop_words

    def is_it_HTML(self,soup):
        test_text = soup.prettify().splitlines()
        test = test_text[0]+' '+test_text[1]+' '+test_text[2]
        p = re.compile('html')
        find = p.search(test)
        if find:
            return True
        else:
            return False


    def remove_soup_tag(self,soup):
        for j in soup(remove_tag_list):
            j.extract()
        return soup


    def get_soup(self, path):
        html = open(path,'r').read()
        soup = BeautifulSoup(html,'html.parser')
        if self.is_it_HTML(soup):
            soup = self.remove_soup_tag(soup)
            return soup
        else:
            return None


    def get_all_title(self,title_list):
        write_path = self.basic_path[5]
        with open(write_path+'title_sentence.txt','w',encoding='UTF-8') as f:
            for title in title_list:
                f.write('%s\n'%title)


    def get_all_time(self,time_list):
        write_path = self.basic_path[5]
        with open(write_path+'published_time.txt','w',encoding='UTF-8') as f:
            for time in time_list:
                f.write('%f\n'%time)


    def get_sorted_file_list(self,name_list,time_list):
        write_path = self.basic_path[5]
        a = time_list
        b = name_list
        a, b = (list(t) for t in zip(*sorted(zip(a,b))))
        with open(write_path+'list_time.txt','w',encoding='UTF-8') as f:
            for name in b:
                f.write('%s\n'%name)


    def make_corpus(self):
        title_list=[]
        doi_list=[]
        year_month_list=[]
        year_list=[]
        month_list=[]
        total_tokens=[]
        total_sents=[]
        sort_file_list=[]
        title_set = set()
        total_corpus = []
        count_files = -1
        for i, path in enumerate(self.path_list):
            soup = self.get_soup(path)
            if not soup:
                print ('%d file from file_list is error!!!!'%(i+1))
                continue
            info = Get_HTML_Information(soup)
            title_sentence = info.get_title_sentence()
            if title_sentence == 'no title':
                print ('%d file from file_list has title error!!!!'%(i+1))

            title_sentence.replace('\n',' ').replace('\r',' ')
            if title_sentence in title_set:
                print ('%d file from file_list has same title!!!!'%(i+1))
                continue
            else:
                count_files = count_files + 1
                sort_file_list.append(self.name_list[i])
                title_set.add(title_sentence)
                title_list.append(title_sentence)

            year_month, year = info.get_pub_year_month()
            year_list.append(year)
            year_month_list.append(year_month)
            month_list.append(round((year_month-year)*12,0))

            raw_text = info.get_abst(title_sentence)
            raw_text = title_sentence+'.'+'\n'+raw_text

            if self.option[0] == True:
                full_text = info.get_full_text()
                write_path = self.basic_path[3]
                with open(write_path+'full_text_%d.txt'%(count_files+1),'w',encoding = 'UTF-8') as f:
                    f.write(full_text)

            if self.option[1] == True:
                write_path = self.basic_path[2]
                with open(write_path+'raw_text_%d.txt'%(count_files+1),'w',encoding = 'UTF-8') as f:
                    f.write('HTML_FILE_NAME\n')
                    f.write('%s\n'%self.name_list[i])
                    f.write('\n')
                    f.write(raw_text)

            if self.option[4] == True:
                rake = Rake("SmartStoplist.txt")
                keywords = rake.run(raw_text)
                write_path = self.basic_path[8]
                with open(write_path+'RAKE_%d.txt'%(count_files+1),'w',encoding = 'UTF-8') as f:
                    if len(keywords) > 4:
                        f.write('%s\n'%keywords[0][0])
                        f.write('%s\n'%keywords[1][0])
                        f.write('%s\n'%keywords[2][0])
                        f.write('%s\n'%keywords[3][0])
                        f.write('%s\n'%keywords[4][0])
                    else:
                        for j in range(len(keywords)):
                            f.write('%s\n'%keywords[j][0])

            write_path = self.basic_path[4]
            sentences = nltk.sent_tokenize(raw_text)
            total_tokens.append([])
            f= open(write_path+'tokens_%d.txt'%(count_files+1),'w',encoding='UTF-8')
            for j in sentences:
                tokens = get_tokens(j)
                pos_token = nltk.pos_tag(tokens)
                filtered_tokens = [word for word in pos_token if not word[0] in self.stop_words]
                lemma_first = lemmatize_tokens_for_pos(filtered_tokens, lemmatizer)
                lemmatized_tokens = [word for word in lemma_first if not word in self.stop_words]
                remove_num = [word for word in lemmatized_tokens if word > 'a' and word.isalnum() and not word[0].isdigit()]
                final_word = []
                for word in remove_num:
                    if len(word) == 2:
                        if not is_useful_2gram(word):
                            continue
                        else:
                            final_word.append(word)
                    else:
                        final_word.append(word)
                #total_tokens[-1] = total_tokens[-1] + lemmatized_tokens
                #total_sents.append(lemmatized_tokens)
                total_tokens[-1] = total_tokens[-1] + final_word
                total_sents.append(final_word)
                for word in final_word:
                    f.write('%s '%(word))
                f.write('\n')
            f.close()


            if len(total_tokens[count_files]) < 30:
                print('Please check %d file => too short!!!'%(i+1))

        for i, text in enumerate(total_tokens):
            sub_corpus = ""
            for words in text:
                sub_corpus = sub_corpus+' '+words
            total_corpus.append(sub_corpus)


        write_path = self.basic_path[5]
        with open(write_path+'embedding.txt','w',encoding='UTF-8') as f:
            for sents in total_sents:
                for words in sents:
                    f.write(words+' ')
                f.write('\n')


        self.get_all_title(title_list)
        self.get_all_time(year_month_list)
        self.get_sorted_file_list(sort_file_list, year_month_list)

        return count_files, total_corpus


    def build_corpus(self,reuse_path, corpus_length):
        if self.option[3] == True:
            total_corpus = []
            for i in range(corpus_length):
                try:
                    with open(reuse_path+'tokens_%d.txt'%(i+1),'r',encoding='UTF-8') as f:
                        total_corpus.append(f.read())
                except IOError:
                    break

        else:
            if self.option[5] or self.option[6]:
                corpus_index, total_corpus_gscholar = self.make_corpus()
            else:
                corpus_index=0
                total_corpus_gscholar=[]

            if self.option[7]:
                PubMed = PubMedTokenizer(corpus_index,'pubmed_result.txt','pubmed_result.xml',self.option, self.basic_path, self.stop_words)
                total_corpus_pubmed = PubMed.run()
            else:
                total_corpus_pubmed = []

            total_corpus = total_corpus_gscholar+total_corpus_pubmed

        return total_corpus


class TFIDF_Vectorizer_scikit:
    def __init__(self,corpus,option,path):
        self.corpus = corpus
        self.option = option
        self.basic_path = path


    def build_vectorizer(self, idf_option):
        vectorizer = TfidfVectorizer(max_df=self.option[2],min_df=self.option[3],norm='l2',ngram_range=self.option[4], use_idf=idf_option)
        matrix = vectorizer.fit_transform(self.corpus)
        self.matrix = matrix
        unique_words = vectorizer.get_feature_names()
        self.words = unique_words
        self.idf = vectorizer.idf_
        print('TFIDF matrix shape is')
        print(matrix.shape)
        print('The number of TFIDF unique word set is')
        print(len(unique_words))
        return matrix, unique_words


    def get_unique_word(self):
        return self.words


    def get_idf_list(self):
        return self.idf


    def get_rank(self, matrix, unique_words):
        Unique_set = set()
        for i, mat in enumerate(matrix):
            col_list = mat.nonzero()[1]
            if len(col_list) == 0:
                continue

            tfidf_list = [matrix[i,col] for col in col_list]
            tfidf_list, col_list = (list(t) for t in zip(*sorted(zip(tfidf_list,col_list),reverse=True)))

            keyword_in_file = []
            new_word_in_file = []
            if self.option[1]:
                write_path = self.basic_path[6]
                with open(write_path+'TFIDF_%d.txt'%(i+1),'w',encoding='UTF-8') as f:
                    f.write('FILENUM=%d\n'%(i+1))
                    f.write('\n\n')
                    for j, col in enumerate(col_list):
                        if unique_words[col] in Unique_set:
                            f.write('%s    %f    %s\n'%(unique_words[col],tfidf_list[j],'old'))
                        else:
                            f.write('%s    %f    %s\n'%(unique_words[col],tfidf_list[j],'new'))
                            new_word_in_file.append(unique_words[col])
                    f.write('\n')

            all_word_in_file = [unique_words[col] for col in col_list]

            if self.option[5]:
                Unique_set = Unique_set|set(all_word_in_file)
            else:
                Unique_set = Unique_set|set(keyword_in_file)


    def df_analyzer(self, word):
        word_index = self.words.index(word)
        word_column = self.matrix[:word_index].nonzero()[0]
        return len(word_column)


    def get_df_list(self):
        df_array = []
        for i in range(len(self.words)):
            df = len(self.matrix[:,i].nonzero()[0])
            df_array.append(df)

        return df_array


    def build_co_occur_matrix(self):
        write_path = self.basic_path[5]
        out_index_list = [100,500,1000,2000]
        mat_out_option = False
        word_num = len(self.words)
        co_matrix = np.zeros((word_num,word_num), dtype=np.int16)
        for i, mat in enumerate(self.matrix):
            col_list = mat.nonzero()[1]
            for j in col_list:
                for k in col_list:
                    co_matrix[j,k] += 1

            if all(((i+1) in out_index_list,mat_out_option)):
                with open(write_path+'occur_mat_%d.txt'%(i+1),'w') as f:
                    for j in range(len(co_matrix)):
                        f.write('\n')
                        for k in range(len(co_matrix)):
                            if j <= k:
                                f.write('%d '%co_matrix[j,k])

        return co_matrix


    def get_equ_index(self, index, column, df_list):
        equ_index_list = (column*column)/column[index]/df_list
        word_index_list = range(len(column))
        equ_index_list, word_index_list = (list(t) for t in zip(*sorted(zip(equ_index_list, word_index_list),reverse=True)))
        pop_index = word_index_list.index(index)
        equ_index_list.pop(pop_index)
        word_index_list.pop(pop_index)
        return equ_index_list, word_index_list


    def get_test_idea(self):
        df_list = self.get_df_list()
        co_matrix = self.build_co_occur_matrix()
        idea_list = []

        #0. Ngram test
        Ngram_check_set = set()
        Ngram_list = self.Ngram_Checker()
        for each_Nword in Ngram_list:
            for i in range(len(each_Nword)-1):
                co_matrix[each_Nword[i]] = 0
                Ngram_check_set.update(each_Nword[i])


        #1. select highest df word and get column
        for i in range(10):         # total 10 ideas
            word_index = df_list.index(max(df_list))
            idea_list.append([])
            idea_list[i].append(word_index)
            column = co_matrix[word_index]
            for j in range(10):      # top 5
                #2. calculate equ_index and select new word
                equ_index_list, equ_word_index_list = self.get_equ_index(word_index, column, df_list)
                while word_index in idea_list[i]:
                    new_index = prob_based_index(equ_index_list[:10])
                    word_index = equ_word_index_list[new_index]
                idea_list[i].append(word_index)
                column = co_matrix[word_index]

        write_path = self.basic_path[5]
        f = open(write_path+'test_new_idea.txt','w')
        for idea in idea_list:
            for index in idea:
                #if index in Ngram_check_set:
                    #idea_word = self.Ngram_converter(Ngram_check_set,word_index)
                #else:
                f.write('%s / '%self.words[index])
            f.write('\n')

        f.close()
        return idea_list


    def get_test_idea2(self):
        KeySet = setlink.KeywordSet(self.words, self.list_for_keywordset(), 1, self.basic_path)
        targetWordlist=['tcnq','conduct']



    def get_std_list(self):
        std_list = []
        co_matrix = self.build_co_occur_matrix()
        for i in range(len(co_matrix)):
            std = np.std(co_matrix[i]/co_matrix[i,i])
            std_list.append(std)

        write_path = self.basic_path[5]
        index_list = range(len(co_matrix))
        std_list, index_list = (list(t) for t in zip(*sorted(zip(std_list,index_list))))
        with open(write_path+'std_list.txt','w') as f:
            for index in index_list:
                f.write('%s\n'%self.words[index])

        return std_list


    def Ngram_Checker(self):
        Ngram_vectorizer = TfidfVectorizer(max_df=1.0,min_df=0.1,norm='l2',ngram_range=(2,4))
        matrix = Ngram_vectorizer.fit_transform(self.corpus)
        unique_words = Ngram_vectorizer.get_feature_names()
        print('The number of Ngram unique word set is')
        print(len(unique_words))
        Ngram_check_list = []
        for Nword in unique_words:
            token = Nword.split()
            sub_list = [(self.words.index(x),self.words.index(y)) for x in token for y in token if x!=y]
            sub_list.append(Nword)
            Ngram_check_list.append(sub_list)

        return Ngram_check_list


    #def Connection_Checker(self):
        #co_matrix = self.build_co_occur_matrix()


    def list_for_keywordset(self, keycut):
        target_list = []
        write_path = self.basic_path[5]
        with open(write_path+'keywords.txt','w') as f:
            f.write('')

        for i, mat in enumerate(self.matrix):
            col_list = mat.nonzero()[1]
            if len(col_list) == 0:
                continue

            tfidf_list = [self.matrix[i,col] for col in col_list]
            tfidf_list, col_list = (list(t) for t in zip(*sorted(zip(tfidf_list,col_list),reverse=True)))
            target_list.append(set(col_list[:keycut]))
            with open(write_path+'keywords.txt','a') as f:
                for i in col_list[:keycut]:
                    f.write('%s '%(self.words[i]))
                f.write('\n')

        return target_list


    def get_keyword_convergence(self, keycut):
        keyword_list = self.list_for_keywordset(keycut)
        total_set = set()
        percent_list = []
        for i, mat in enumerate(self.matrix):
            col_list = mat.nonzero()[1]
            percent_list.append(len(keyword_list[i]&total_set)/len(keyword_list[i]))
            total_set = total_set|set(col_list)

        return percent_list

    def run(self):
        matrix, unique_words = self.build_vectorizer(True)
        #self.matrix = matrix
        self.get_rank(matrix,unique_words)


    def get_norm_tf(self):
        matrix, unique_words = self.build_vectorizer(False)
        return matrix



class ONLYTF_Vectorizer_scikit(TFIDF_Vectorizer_scikit):

    def build_vectorizer(self):
        vectorizer = CountVectorizer(max_df=self.option[2],min_df=self.option[3],ngram_range=self.option[4])
        matrix = vectorizer.fit_transform(self.corpus)
        self.matrix = matrix
        unique_words = vectorizer.get_feature_names()
        self.words = unique_words
        print('ONLYTF matrix shape is')
        print(matrix.shape)
        print('The number of ONLYTF unique word set is')
        print(len(unique_words))
        return matrix, unique_words


    def get_rank(self, matrix, unique_words):
        Unique_set = set()
        for i, mat in enumerate(matrix):
            col_list = mat.nonzero()[1]
            if len(col_list) == 0:
                continue

            tfidf_list = [matrix[i,col] for col in col_list]
            tfidf_list, col_list = (list(t) for t in zip(*sorted(zip(tfidf_list,col_list),reverse=True)))

            keyword_in_file = []
            new_word_in_file = []
            if self.option[1]:
                write_path = self.basic_path[7]
                with open(write_path+'ONLYTF_%d.txt'%(i+1),'w',encoding='UTF-8') as f:
                    f.write('FILENUM=%d\n'%(i+1))
                    f.write('\n\n')
                    for j, col in enumerate(col_list):
                        if unique_words[col] in Unique_set:
                            f.write('%s    %f    %s\n'%(unique_words[col],tfidf_list[j],'old'))
                        else:
                            f.write('%s    %f    %s\n'%(unique_words[col],tfidf_list[j],'new'))
                            new_word_in_file.append(unique_words[col])
                    f.write('\n')

            all_word_in_file = [unique_words[col] for col in col_list]

            if self.option[5]:
                Unique_set = Unique_set|set(all_word_in_file)
            else:
                Unique_set = Unique_set|set(keyword_in_file)


    def run(self):
        matrix, unique_words = self.build_vectorizer()
        #self.matrix = matrix
        self.get_rank(matrix,unique_words)


class Get_Recall_Precision:
    """ f measure class """
    def __init__(self, total, manual, computation):
        self.manual = maunal
        self.computation = computation


    #def word_stemming(self):



    #def word_lemma(self):



    def get_TP_TN(self, manual, computation):
        man_set = set(manual)
        com_set = set(computation)

        TP = len(man_set&com_set)
        FP = len(com_set-man_set)
        FN = len(man_set-com_set)
        return TP,TN,FP,FN


    def get_precision(self, word_set):
        precision = word_set[0]/(word_set[0]+word_set[1])
        return precision


    def get_recall(self, word_set):
        recall = word_set[0]/(word_set[0]+word_set[2])
        return recall


    def get_f_score(self, word_set):
        p = word_set[0]/(word_set[0]+word_set[1])
        r = word_set[0]/(word_set[0]+word_set[2])
        f_score = 2*p*r/(p+r)
        return f_score


    def run(self):
        lemmas = self.word_lemma()
        TP_TN_FP_FN = self.get_TP_TN(lemmas)

        precision = self.get_precision(TP_TN_FP_FN)
        recall = self.get_recall(TP_TN_FP_FN)
        #accuracy = self.get_accuracy(TP_TN_FP_FN)
        f_score = self.get_f_score(TP_TN_FP_FN)

        return precision, recall, f_score



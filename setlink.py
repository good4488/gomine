import re
import numpy

class KeywordSet:
    def __init__(self, word_list, target_list, option, basic_path):
        self.target = target_list
        self.words = word_list
        self.option = option
        self.basic_path = basic_path


    def get_total_keywordSet(self):
        total_set = set()
        for tar_set in self.target:
            total_set = total_set|tar_set
        return total_set


    def make_intersection_map(self):
        inter_map = dict()
        for i in range(len(self.target)):
            i_set = self.target[i]
            for j in range(i+1,len(self.target)):
                j_set = self.target[j]
                inter_set = i_set & j_set
                inter_map[(i,j)] = len(inter_set)
        return inter_map


    def get_intersection(self, i_j):
        inter_set = self.target[i_j[0]] & self.target[i_j[1]]
        return inter_set


    def get_intersection_byfileindex(self, indexlist):
        inter_set = self.target[indexlist[0]]
        for index in indexlist:
            inter_set = inter_set & self.target[index]
        return inter_set


    def get_interfile_byword(self, wordlist):
        try:
            word_index_set = set([self.words.index(word) for word in wordlist])
        except ValueError:
            print('word does not exist!')
            return None
        else:
            target_index = [self.target.index(sets) for sets in self.target if not (word_index_set-sets)]
            return target_index


    def get_interfile_byregex(self, patternlist):
        p_list = [r'^.*'+pattern+'.*$' for pattern in patternlist]
        target_index = []
        for i, sets in enumerate(self.target):
            find_list = [p for p in p_list if any([re.search(p,self.words[word]) for word in sets])]
            if len(find_list) == len(p_list):
                target_index.append(i)
        return target_index


#def estimate_Citation(


    def run(self):
        print('Num of total Keywords')
        print(len(self.get_total_keywordSet()))
        intersection_map = self.make_intersection_map()
        inter_key = list(intersection_map.keys())
        inter_value = list(intersection_map.values())
        inter_value, inter_key = (list(t) for t in zip(*sorted(zip(inter_value,inter_key), reverse=True)))
        for key in inter_key[:30]:
            inter_set = self.get_intersection(key)
            print('%d file %d file'%(key[0]+1,key[1]+1))
            inter_keyword_list = [self.words[index] for index in list(inter_set)]
            print(inter_keyword_list)
            print('\n')

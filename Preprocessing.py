# -*- coding:utf-8 -*-
import re
import nltk
import string
import codecs

from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag


class Get_HTML_Information:
    """get HTML datas like (1) published year,month (2) title (3) abstract (4) full_text (5) only_text"""

    def __init__(self, soup):
        self.soup = soup


    def get_pub_year(self):
        """ get publish year from HTML soup"""
        pattern_date_value = re.compile('year|date|issue',re.I)
        pattern_day_month_year = re.compile('(\d+)/(\d+)/(\d+)')
        pattern_month = re.compile('January|February|March|April|May|June|July|August|September|October|November|December|january|february|march|april|june|july|august|september|october|november|december')
        pattern_year = re.compile('(\d{4})')
        date = ""
        int_date = 0
        date_group = []
        for i in range(len(self.soup.find_all('meta'))):
            list_value = list(self.soup.find_all('meta')[i].attrs.values())
            list_key = list(self.soup.find_all('meta')[i].attrs.keys())
            for j in range(len(list_value)):
                value_find = pattern_date_value.search(list_value[j])
                if value_find and list_key[j]=='name':
                    date_find = pattern_day_month_year.search(self.soup.find_all('meta')[i].attrs['content'])
                    if date_find:
                        for k in range(1,4):
                            if len(date_find.group(k)) == 4:
                                int_date = int(date_find.group(k))
                                date_group.append(int_date)
                    break
        if date_group:
            return min(date_group)
        else:
            if self.soup.body:
                each_sentence= self.soup.body.get_text().split()
            else:
                each_sentence= self.soup.get_text().split()
                print('soup error')
            for j in range(len(each_sentence)):
                month_find = pattern_month.search(each_sentence[j])
                if month_find:
                    before_find = pattern_year.search(each_sentence[j-1])
                    after_find = pattern_year.search(each_sentence[j+1])
                    if after_find:
                        date_group.append(int(after_find.group(0)))
                    elif before_find:
                        date_group.append(int(before_find.group(0)))
                    else:
                        before_find = pattern_year.search(each_sentence[j-2])
                        after_find = pattern_year.search(each_sentence[j+2])
                        if after_find:
                            date_group.append(int(after_find.group(0)))
                        elif before_find:
                            date_group.append(int(before_find.group(0)))

            if date_group:
                return min(date_group)
            else:
                pattern_month = re.compile('Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec')       #### 'may'
                for j in range(len(each_sentence)):
                    month_find = pattern_month.search(each_sentence[j])
                    if month_find:
                        before_find = pattern_year.search(each_sentence[j-1])
                        after_find = pattern_year.search(each_sentence[j+1])
                        if after_find:
                            date_group.append(int(after_find.group(0)))
                        elif before_find:
                            date_group.append(int(before_find.group(0)))
                        else:
                            before_find = pattern_year.search(each_sentence[j-2])
                            after_find = pattern_year.search(each_sentence[j+2])
                            if after_find:
                                date_group.append(int(after_find.group(0)))
                            elif before_find:
                                date_group.append(int(before_find.group(0)))
                if date_group:
                    return min(date_group)
                else:
                    return 1234


    def get_pub_year_month(self):
        pattern_day_month_year = re.compile('(\d+)/(\d+)/(\d+)')
        pattern_day_month_year_2 = re.compile('(\d+)-(\d+)-(\d+)')
        pattern_date_value = re.compile('year|date|issue',re.I)
        pattern_month = re.compile('January|February|March|April|May|June|July|August|September|October|November|December|january|february|march|april|june|july|august|september|october|november|december')
        pattern_month_2 = re.compile('Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec') 
        pattern_year = re.compile('(\d{4})')
        pattern_pub = re.compile('publish|publicat')
        month_list = ['january','february','march','april','may','june','july','august','september','october','november','december']
        month_list_2 = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
        date_group=[]
        float_date=0.0

        #from meta tag
        for i in self.soup.find_all('meta'):
            list_value = list(i.attrs.values())
            list_key = list(i.attrs.keys())
            for j in range(len(list_value)):
                value_find = pattern_date_value.search(list_value[j])
                if value_find and list_key[j]=='name':
                    date_find = pattern_day_month_year.search(i.attrs['content'])
                    month_find = pattern_month.search(i.attrs['content'])
                    month_2_find = pattern_month_2.search(i.attrs['content'])
                    if not date_find:
                        date_find = pattern_day_month_year_2.search(i.attrs['content'])
                    if date_find:
                        #print('date find')
                        for k in range(1,4):
                            if len(date_find.group(1)) == 4:                        # Ambiguous month and date XXXX/10/11
                                float_date = int(date_find.group(1)) + (int(date_find.group(2))-1)/12
                                date_group.append(round(float_date,2))
                            elif len(date_find.group(3)) == 4:
                                if int(date_find.group(1)) > 12 and int(date_find.group(2)) < 13:
                                    float_date = int(date_find.group(3)) + (int(date_find.group(2))-1)/12
                                    date_group.append(round(float_date,2))
                                elif int(date_find.group(2)) > 12 and int(date_find.group(1)) < 13:
                                    float_date = int(date_find.group(3)) + (int(date_find.group(1))-1)/12
                                    date_group.append(round(float_date,2))
                                else:
                                    float_date = int(date_find.group(3)) + (int(date_find.group(2))-1)/12
                                    date_group.append(round(float_date,2))
                            else:
                                float_date = int(date_find.group(2)) + (int(date_find.group(1))-1)/12
                        return min(date_group), int(min(date_group))

                    elif month_find:
                        #print('month_find')
                        for k in range(12):
                            if month_list[k] == month_find.group(0).lower():
                                month_index = k
                        token = i.attrs['content'].split()
                        token.sort()
                        if len(token) == 3 and token[0].isdigit() and token[1].isdigit():
                            for word in token:
                                if len(word) == 4 and word.isdigit():
                                    float_date = int(word) + month_index/12
                                    date_group.append(round(float_date,2))
                            return min(date_group), int(min(date_group))

                    elif month_2_find:
                        #print('month2_find')
                        for k in range(12):
                            if month_list_2[k] == month_2_find.group(0).lower():
                                month_index = k
                        token = i.attrs['content'].split()
                        token.sort()
                        if len(token) == 3 and token[0].isdigit() and token[1].isdigit():
                            for word in token:
                                if len(word) == 4 and word.isdigit():
                                    float_date = int(word) + month_index/12
                                    date_group.append(round(float_date,2))
                            return min(date_group), int(min(date_group))

                    else:
                        #print('year find')
                        year_find = pattern_year.search(i.attrs['content'])
                        if year_find and len(i.attrs['content']) == 4:
                            float_date = int(i.attrs['content']) + 0.0
                            date_group.append(round(float_date,2))

        if date_group:
            return min(date_group), int(min(date_group))

        # from pub date sentence
        else:
            if self.soup.body:
                each_sentence= self.soup.body.get_text().split()
            else:
                each_sentence= self.soup.get_text().split()
            for j in range(len(each_sentence)):
                month_find = pattern_month.search(each_sentence[j])
                month_index = -1
                if month_find:
                    for k in range(len(month_list)):
                        if month_list[k] == month_find.group(0).lower():
                            month_index = k
                    before_find = pattern_year.search(each_sentence[j-1])
                    after_find = pattern_year.search(each_sentence[j+1])
                    if after_find:
                        float_date = int(after_find.group(0)) + month_index/12
                        date_group.append(round(float_date,2))
                    elif before_find:
                        float_date = int(before_find.group(0)) + month_index/12
                        date_group.append(round(float_date,2))
                    else:
                        before_find = pattern_year.search(each_sentence[j-2])
                        after_find = pattern_year.search(each_sentence[j+2])
                        if after_find:
                            float_date = int(after_find.group(0)) + month_index/12
                            date_group.append(round(float_date,2))
                        elif before_find:
                            float_date = int(after_find.group(0)) + month_index/12
                            date_group.append(round(float_date,2))

            if date_group:
                return min(date_group), int(min(date_group))

            else:
                for j in range(len(each_sentence)):
                    month_find = pattern_month_2.search(each_sentence[j])
                    month_index = -1
                    if month_find:
                        for k in range(len(month_list_2)):
                            if month_list_2[k]==month_find.group(0).lower():
                                month_index = k
                        before_find = pattern_year.search(each_sentence[j-1])
                        after_find = pattern_year.search(each_sentence[j+1])
                        if after_find:
                            float_date = int(after_find.group(0)) + month_index/12
                            date_group.append(round(float_date,2))
                        elif before_find:
                            float_date = int(before_find.group(0)) + month_index/12
                            date_group.append(round(float_date,2))
                        else:
                            before_find = pattern_year.search(each_sentence[j-2])
                            after_find = pattern_year.search(each_sentence[j+2])
                            if after_find:
                                float_date = int(after_find.group(0)) + month_index/12
                                date_group.append(round(float_date,2))
                            elif before_find:
                                float_date = int(after_find.group(0)) + month_index/12
                                date_group.append(round(float_date,2))

                    if date_group:
                        return min(date_group), int(min(date_group))
                    else:
                        return 1234.0, 1234


    def get_title_sentence(self):
        """get title sentence from HTML meta, title tag"""
        pattern_title = re.compile('title',re.I)
        title = ""
        for i in range(len(self.soup.find_all('meta'))):
            list_value = list(self.soup.find_all('meta')[i].attrs.values())
            list_key = list(self.soup.find_all('meta')[i].attrs.keys())
            for j in range(len(list_value)):
                title_find = pattern_title.search(list_value[j])
                if title_find and list_key[j]=='name':
                    if title:
                        if len(title) < len(self.soup.find_all('meta')[i].attrs['content']):
                            title = self.soup.find_all('meta')[i].attrs['content']
                    else:
                        title = self.soup.find_all('meta')[i].attrs['content']
                    break
        if title:
            return title
        elif not self.soup.find_all('title') or not self.soup.find_all('title')[0].contents:
            title = 'no title'
            return title
        else:
            title = self.soup.find_all('title')[0].contents[0].strip()
            pattern_journal = re.compile('- [A-Z]{1}[a-z]+')
            journal_find = pattern_journal.findall(title)
            if journal_find:
                pattern_journal = re.compile(journal_find[-1])
                journal_find = pattern_journal.search(title)

                title = title[0:journal_find.start()]
                return title
            else:
                return title


    def get_only_text(self):
        for j in self.soup(['title','button','table']):
            j.extract()

        if self.soup.body:
            text = self.soup.body.get_text()
        else:
            text = self.soup.get_text()
        #text_lines = sent_tokenize(text)
        text_lines = text.splitlines()
        text_line_word = []
        for i in text_lines:
            if len(i) > 3:
                i.strip()
                text_line_word.append(i.split())
            else:
                continue

        full_text = ""
        for i in range(len(text_line_word)):
            for j in range(len(text_line_word[i])):
                full_text=full_text+" "+text_line_word[i][j]
                full_text.lstrip()
            full_text=full_text+"\n"
        return full_text


    def get_full_text(self):
        text = self.soup.prettify()
        return text


    def get_abst(self,title):
        tag_list = ['sub','sup']
        for tag in tag_list:
            sub_list = self.soup.find_all(tag,string=True)
            for sub in sub_list:
                sub.replace_with(sub.string.strip()+'qorwns')

        #if self.soup.body:
            #text = self.soup.body.get_text()
            #text = re.sub(r'\n\s*\n', r'\n', self.soup.body.get_text().strip(), flags=re.M)
        #else:
            #text = self.soup.get_text()
        text = re.sub(r'\n\s*\n', r'\n', self.soup.get_text().strip(), flags=re.M)
        list = text.splitlines()
        title_pre = title[:10]

        #### remove before abstract
        full_text=""
        for i in range(len(list)):
            list[i] = list[i].lstrip()
            if list[i]:
                if list[i][-1] == " ":
                    full_text = full_text+''+list[i]
                    if list[i][-6:] == 'qorwns':
                        full_text = full_text[:-6]
                elif list[i][-6:] == 'qorwns':
                    full_text = full_text+''+list[i][:-6]
                elif list[i-1][-6:] == 'qorwns' and len(list[i])==1:
                    full_text = full_text+''+list[i]
                else:
                    full_text = full_text+' '+list[i]

        index = full_text.find(title_pre)
        if index == -1:
            full_text = full_text
        else:
            full_text = full_text[index:]

        index = full_text.find('Abstract')
        if index == -1:
            index = full_text.find('abstract')

        if index == -1:
            pattern_pub_date = re.compile('publish|publicat',re.I)
            pub_suffix=""
            for i in range(len(list)):
                if list[i]:
                    pub_find = pattern_pub_date.search(list[i])
                    if pub_find:
                        pub_index = list[i].find(pub_find.group(0))
                        pub_suffix = list[i][pub_index:-1]
                        break
                    else:
                        pub_suffix = ""
                else:
                    pub_suffix = ""

            index = full_text.find(pub_suffix)

            if not pub_suffix:
                full_text = full_text
            else:
                full_text = full_text[index+len(pub_suffix)+1:]

        else:
            full_text = full_text[index+8:]

        #### removed after conclusion

        head_list = ['Acknowled','acknowled','Reference','Copyright','copyright','Advertisement','COLLAPSE','Article Information', 'Cookies']
        index_list = []
        for i in head_list:
            index = full_text.find(i)
            if index > 0:
                index_list.append(index)

        if index_list:
            index = min(index_list)
        else:
            index = -1
            #print('error\n')
        full_text = full_text[:index]

        return full_text



def get_tokens(text):
    #p = re.compile('\(.*\)|\[.*\]|\{.*\}')
    lowers = text.lower()
    #lowers = p.sub(' ', lowers)
    #no_punct = lowers.translate(str.maketrans(',-−——­→—×≪‖⊥∼〉〈≤≥→"′‘≈“”&\'()≡+:;<=>_`{|}~·––/',"                                             "))
    no_punct = lowers.translate(str.maketrans(',→×≪‖⊥∼〉〈≤≥→"′‘≈“”&\'≡+:;<=>_`{|}~·/',"                                   ",'-−——­—––'))
    no_punct = no_punct.translate(str.maketrans("","", '!©∧↑χσηϕμτθ∞φ∑()γλ†#±⋯$δ°⋅β%α*.á?@\\^âåå[]'+string.punctuation ))
    #no_punct = lowers.translate(str.maketrans("","", '!×©∧↑χ≪ση–‖ϕ⊥μ∼τθ〉∞φ〈≤∑≥γλ—→†"′#‘⋯$δ°≈⋅“”β%≡&\'()α*+,-.á:;<=>?@\\^_`{|}~−·âåå­'+string.punctuation ))
    #no_punct = lowers.translate(str.maketrans("","", '!×©∧↑χ≪ση–‖ϕ⊥μ∼τθ〉∞φ〈≤∑≥γλ—→†"′#‘⋯$δ°≈⋅“”β%≡&\'()α*+,-.á:;<=>?@\\^_`{|}~−·âåå­'+string.punctuation ))
    #no_punct = lowers.translate(str.maketrans("","", '!×©∧↑χ≪ση–‖ϕ⊥μ∼τθ〉∞φ〈≤∑≥γλ—→†"′#‘⋯$δ°≈⋅“”β%≡&\'()α*+,-.á:;<=>?@\\^_`{|}~−·âåå­'+string.punctuation ))

    tokenizer = RegexpTokenizer('\s+',gaps=True)
    tokens = tokenizer.tokenize(no_punct)

    real_tokens = []
    for item in tokens:
        if item.isdigit():
            continue
        else:
            if item.isalnum:
                real_tokens.append(item)

    return real_tokens


def lemmatize_tokens(tokens, lemmatizer):
    lemma = []
    for item in tokens:
        lemma.append(lemmatizer.lemmatize(item))
    return lemma


#def stem_tokens():





def lemmatize_tokens_for_pos(tokens, lemmatizer):
    lemma = []
    for word, pos in tokens:
        tag = pos[0].lower()
        tag = tag if tag in ['a','r','n','v'] else None
        if not tag:
            lemma.append(word)
        #elif tag in ['a', 'r']:
            #continue
        else:
            lemma.append(lemmatizer.lemmatize(word, tag))

    return lemma

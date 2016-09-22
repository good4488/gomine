from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from Preprocessing import *

lemmatizer_pubmed = WordNetLemmatizer()


def convert_date(date):
    date_list = date.split()
    year = int(date_list[0])
    month_list = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    if len(date_list) == 1:
        month = 6
    elif date_list[1].lower() in month_list:
        month = month_list.index(date_list[1].lower()) + 1
    else:
        month = 6
    return (year,month)


def pubmed_keywords(path):
    import xml.etree.ElementTree as ET
    doc = ET.parse('pubmed_result.xml')
    root = doc.getroot()

    for child in root.iter('PubmedArticle'):
        for c in child.iter('KeywordList'):
            print (c.findtext('Keyword'))


class PubMedTokenizer():
    def __init__(self, corpus_index, text_path, xml_path, option, basic_path, stop_words):
        self.corpus_index = corpus_index
        self.text_path = text_path
        self.xml_path = xml_path
        self.basic_path = basic_path
        self.option = option
        self.stop_words = stop_words
        print('PubMed used')
        print('PubMed starting index = %d'%(corpus_index))

    def get_keyword_list(self):
        import xml.etree.ElementTree as ET
        doc = ET.parse(self.xml_path)
        root = doc.getroot()
        keyword_list = []

        for child in root.iter('PubmedArticle'):
            sub_list = []
            for c in child.iter('KeywordList'):
                for d in c.iter('Keyword'):
                    if d.text:
                        sub_list.append(d.text.lower())

            keyword_list.append(sub_list)
        return keyword_list

    def text_loader(self):
        with open(self.text_path,'r',encoding='UTF-8') as f:
            line_text = f.read().split('\n\n')[:-1]

        return line_text


    def line_to_structed_data(self):
        line_text = self.text_loader()
        raw_text_list = []
        doi_list = []
        PMID_list = []
        title_list = []
        journal_list = []
        date_list = []
        keyword_list = self.get_keyword_list()
        file_count = 0
        for i, line in enumerate(line_text):
            if line[0] == '\n' and line[1].isdigit():
                if i != 0:
                    DOI_data = line_text[i-1].split()
                    if 'DOI:' in DOI_data:
                        index = DOI_data.index('DOI:')
                        doi_list.append(DOI_data[index+1])
                    else:
                        doi_list.append('NO DOI data')
                    if 'PMID:' in DOI_data:
                        index = DOI_data.index('PMID:')
                        PMID_list.append(DOI_data[index+1])
                    else:
                        PMID_list.append('NO PMID data')

                file_count += 1
                line_data = line.split('.')
                journal_list.append(line_data[1].strip())

                raw_date = line_data[2].split(';')[0].strip()
                date = convert_date(raw_date)
                date_list.append(date)

                title = ""
                raw_title = line_text[i+1].lower().split()
                for token in raw_title:
                    title = title +' '+ token
                title_list.append(title[1:])

                if line_text[i+3][:6] == 'Author':
                    raw_text = line_text[i+4].lower().split()
                    text = ''
                    for token in raw_text:
                        text = text+' '+token
                else:
                    raw_text = line_text[i+3].lower().split()
                    text = ''
                    for token in raw_text:
                        text = text+' '+token

                raw_text_list.append(text[1:])

            if i == (len(line_text)-1):
                DOI_data = line_text[i-1].split()
                if 'DOI:' in DOI_data:
                    index = DOI_data.index('DOI:')
                    doi_list.append(DOI_data[index+1])
                else:
                    doi_list.append('NO DOI data')
                if 'PMID:' in DOI_data:
                    index = DOI_data.index('PMID:')
                    PMID_list.append(DOI_data[index+1])
                else:
                    PMID_list.append('NO PMID data')

        return raw_text_list, doi_list, PMID_list, title_list, journal_list, date_list, keyword_list


    def make_output(self, index, structed_list):
        #write_path = self.basic_path[5]
        write_path=self.basic_path[5]
        if index == 0:
            filename = 'pubmed_doi.txt'
            with open(write_path+filename,'w') as f:
                for data in structed_list:
                    f.write('%s\n'%data)
        if index == 1:
            filename = 'pubmed_PMID.txt'
            with open(write_path+filename,'w') as f:
                for data in structed_list:
                    f.write('%s\n'%data)
        if index == 2:
            filename = 'pubmed_titles.txt'
            with open(write_path+filename,'w') as f:
                for data in structed_list:
                    f.write('%s\n'%data)
        if index == 3:
            filename = 'pubmed_journals.txt'
            with open(write_path+filename,'w') as f:
                for data in structed_list:
                    f.write('%s\n'%data)
        if index == 4:
            filename = 'pubmed_date.txt'
            with open(write_path+filename,'w') as f:
                for year, month in structed_list:
                    f.write('%s %s\n'%(year,month))


    def get_pubmed_tokens(self, title_list, raw_text_list):
        total_corpus = []
        total_tokens =[]
        total_sents = []
        for i, text in enumerate(raw_text_list):
            new_text = title_list[i]+'\n'+text
            sentences = nltk.sent_tokenize(new_text)
            total_tokens.append([])
            write_path=self.basic_path[4]
            f= open(write_path+'tokens_%d.txt'%(self.corpus_index+i+1),'w',encoding='UTF-8')
            for sent in sentences:
                tokens = get_tokens(sent)
                pos_token = nltk.pos_tag(tokens)
                filtered_tokens = [word for word in pos_token if not word[0] in self.stop_words]
                lemma_first = lemmatize_tokens_for_pos(filtered_tokens, lemmatizer_pubmed)
                lemmatized_tokens = [word for word in lemma_first if not word in self.stop_words]
                remove_num = [word for word in lemmatized_tokens if word > 'a' and word.isalnum() and not word[0].isdigit()]
                final_word = []
                for word in remove_num:
                    if len(word) == 2:
                        #if not is_useful_2gram(word):
                            #continue
                        #else:
                            #final_word.append(word)
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

        for i, text in enumerate(total_tokens):
            sub_corpus=''
            for words in text:
                sub_corpus = sub_corpus+' '+words
            total_corpus.append(sub_corpus)

        write_path = self.basic_path[5]
        with open(write_path+'embedding.txt','a+',encoding='UTF-8') as f:
            for sents in total_sents:
                for words in sents:
                    f.write(words+' ')
                f.write('\n')

        return total_corpus


    def run(self):
        raw_text_list, doi_list, PMID_list, title_list, journal_list, date_list, keyword_list = self.line_to_structed_data()
        token_list = self.get_pubmed_tokens(title_list, raw_text_list)

        for i, data in enumerate([doi_list, PMID_list, title_list, journal_list, date_list]):
            self.make_output(i,data)

        return token_list


#class ManualKeywordAnalyzer():


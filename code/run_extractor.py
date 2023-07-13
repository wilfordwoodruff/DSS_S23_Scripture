#%%
import pandas as pd
from StringUtil import StringUtil
from MatchExtractor import MatchExtractor
import warnings

pd.set_option('display.max_colwidth', None)
warnings.filterwarnings('ignore')

replacements_woodruff = {
    r'(, b\.)'              : r'',
    r'\<U\+25CA\>'          : r'',
    r'\&amp;c?'             : r"and",
    r'\&apos;'              : r"'",
    r"(\^?FIGURES?\^?)"     : r'',
    r'[\{\}\~]'             : r'',
    r'\n'                   : r' ',
    r'\[\[(.*?)\|(.*?)\]\]' : r'\1',
    r'\n'                   : r' ',
    r'\–|\-|\—|\-\s'        : r'',
    r'\s+'                  : r' ',
    r'\.|\:|\;|\,|\(|\)|\?' : r'',
    r'confer ence|Conferance'     : r'conference',
    r'sacrafice'           : r'sacrifice',
    r'discours'            : r'discourse',
    r'travling'            : r'traveling',
    r'oclock'              : r'oclock',
    r'w\. woodruff'        : r'wilford woodruff',
    r'any\s?whare'         : r'anywhere',
    r'some\s?whare'        : r'somewhere',
    r'whare'               : r'where',
    r'sumthing'            : r'something',
    r' els '               : r' else ',
    r' wil '               : r' will ',
    r'savio saviour'     : r'saviour',
    r'arived'            : r'arrived',
    r'intirely    '      : r'entirely',
    r'phylosophers'      : r'philosophers',
    r'baptised'           : r'baptized',
    r'benef\- it'       : r'benefit',
    r'preachi \-ng'      : r'preaching',
    r'oppor- tunities' : r'opportunities',
    r'vary'         : r'very',
    r'councellor'   : r'counselor',
    r'sircumstances' : r'circumstances',
    r'Preasent'    : r'present',
    r'sept\.'      : r'september',
    r'sacramento sacramento' : r'sacramento',
    r'tryed'       : r'tried',
    r'fals'        : r'false',
    r'aprail'      : r'april',
    r'untill'      : r'until',
    r'sumwhat'      : r'somewhat',
    r'joseph smith jun' : r'joseph smith jr',
    r'miricle' : r'miracle',
    r'procedings' : r'proceedings',
    r'w odruff' : r'woodruff',
    r'prefered' : r'preferred',
    r'esspecially' : r'especially',
    r'ownly' : r'only',
    r'th\[e\]' : r'the',
    r'judjment' : r'judgement',
    r'experiance' : r'experience',
    r'ingaged' : r'engaged',
    r'\[she\]' : r'she',
    r'fulnes ' : r'fulness ',
    r'interestin ' : r'interesting ',
    r'respetible ' : r'respectable ',
    r'attonement' : r'atonement',
    r'diestroy ' : r'destroy ',
    r'a b c d e f g h i j k l m n o p q r s t u v w x y z and 1 2 3 4 5 6 7 8 9 0' : r'',
    r' \^e\^ 4 \^p\^ 5 \^t\^ 1 \^d\^ 3 ': r'',
    r'W X Y Z and 1 2 3 4 5': r'',
}

scripture_replacements = {
    r'\.|\:|\;|\,|\-|\(|\)|\?' : r'',
}

#%%
path_root = '../data/all_books_10_words/'
path_data_woodruff_raw   = '../data/data_woodruff_raw.csv'
# path_data_woodruff_clean = path_root + 'data_woodruff_clean.csv'
path_data_scriptures     = '../data/data_scriptures.csv'

# url paths
url_woodruff = "https://github.com/wilfordwoodruff/Main-Data/raw/main/data/derived/derived_data.csv"
url_scriptures = 'https://github.com/wilfordwoodruff/wilford_woodruff_hack23/raw/main/data/lds-scriptures.csv'

# load data
data_scriptures = pd.read_csv(path_data_scriptures)
data_woodruff = pd.read_csv(path_data_woodruff_raw)

# clean woodruff data
data_woodruff['text'] = StringUtil.str_replace_column(data_woodruff['text'], replacements_woodruff)
# clean scripture data
data_scriptures['text'] = StringUtil.str_replace_column(data_scriptures['text'], scripture_replacements)

# filter to certain volumes
volume_titles = [
     'Old Testament',
     'New Testament',
     'Book of Mormon',
     'Doctrine and Covenants',
     'Pearl of Great Price',
     ]
data_scriptures = data_scriptures.query("volume_title in @volume_titles")
# query = "verse_title == 'Doctrine and Covenants 136:11'|verse_title == 'Doctrine and Covenants 136:12'|verse_title == 'Doctrine and Covenants 136:13'|verse_title == 'Doctrine and Covenants 136:14'|verse_title == 'Doctrine and Covenants 136:15'|verse_title == 'Doctrine and Covenants 136:16'|verse_title == 'Doctrine and Covenants 136:17'"
# data_scriptures = data_scriptures.query(query)
data_scriptures

#%%


phrase_length = 10
threshold = .70
print('volumes:', volume_titles)
print('phrase length:', phrase_length)
print('threshold:', threshold)
match_extractor = MatchExtractor(data_woodruff.copy(),
                                 data_scriptures.copy(),
                                 phrase_length,
                                 threshold=threshold,
                                 path_root=path_root)

# iterate through each row of scripture phrases dataset and run TFIDF model and cosine similarity.
match_extractor.run_extractor(extensions=True, save=False, quarto_publish=False)

match_extractor.matches_total

#%%
path = '../data/all_books_13_words/data_matches_extensions_temporary.csv'
data = pd.read_csv(path)
data_scriptures1 = data_scriptures[['verse_title','volume_title']]
data_scriptures1

merged_data = pd.merge(data, data_scriptures1, on='verse_title', how='left')
merged_data.to_csv(path, index = False)


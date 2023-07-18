#%%
import pandas as pd
from StringUtil import StringUtil
from MatchExtractor import MatchExtractor
import warnings

pd.set_option('display.max_colwidth', None)
# warnings.filterwarnings('ignore')

replacements_woodruff = {
    r'\[\[(.*?)\|(.*?)\]\]' : r'\1',
}

#%%
path_root = '../data/matches/all_books_10_words/'
path_data_woodruff_raw   = '../data/raw/data_woodruff_raw.csv'
path_data_woodruff_raw   = '../data/raw/derived_data.csv'
# path_data_woodruff_clean = path_root + 'data_woodruff_clean.csv'
path_data_scriptures     = '../data/raw/data_scriptures.csv'
path_matches = '../data/matches/data_matches2.csv'

# url paths
url_woodruff = "https://github.com/wilfordwoodruff/Main-Data/raw/main/data/derived/derived_data.csv"
url_scriptures = 'https://github.com/wilfordwoodruff/wilford_woodruff_hack23/raw/main/data/lds-scriptures.csv'

# load data
data_scriptures = pd.read_csv(path_data_scriptures)
data_woodruff = pd.read_csv(path_data_woodruff_raw)

# clean woodruff data
columns = ['Internal ID', 'Parent ID', 'Order', 'Document Type', 'Website URL', 'Dates', 'Text Only Transcript']
new_columns = {'Internal ID':'internal_id',
               'Parent ID':'parent_id',
               'Order':'order',
               'Document Type':'document_type',
               'Website URL':'website_url',
               'Dates':'dates',
               'Text Only Transcript':'text_woodruff'
               }
data_woodruff = data_woodruff.rename(columns=new_columns)[list(new_columns.values())]
data_woodruff = data_woodruff.query("document_type=='Journals'")
# text = StringUtil.combine_rows(data_woodruff['text'])
data_woodruff['text_woodruff'] = StringUtil.str_replace_column(data_woodruff['text_woodruff'], replacements_woodruff)
# data_woodruff.info()

#%%
# clean scripture data
data_scriptures = data_scriptures.rename(columns={'text':'text_scriptures'})
# data_scriptures['text_scriptures'] = StringUtil.str_replace_column(data_scriptures['text_scriptures'], scripture_replacements)

# filter to certain volumes
volume_titles = [
    #  'Old Testament',
    #  'New Testament',
    #  'Book of Mormon',
     'Doctrine and Covenants',
    #  'Pearl of Great Price',
     ]
data_scriptures = data_scriptures.query("volume_title in @volume_titles")
# query = "verse_title == 'Doctrine and Covenants 136:11'|verse_title == 'Doctrine and Covenants 136:12'|verse_title == 'Doctrine and Covenants 136:13'|verse_title == 'Doctrine and Covenants 136:14'|verse_title == 'Doctrine and Covenants 136:15'|verse_title == 'Doctrine and Covenants 136:16'|verse_title == 'Doctrine and Covenants 136:17'"
# data_scriptures = data_scriptures.query(query)
data_scriptures

#%%

phrase_length = 10
threshold = .7
print('volumes:', volume_titles)
print('phrase length:', phrase_length)
print('threshold:', threshold)
match_extractor = MatchExtractor(data_woodruff.copy(),
                                 data_scriptures.copy(),
                                 phrase_length,
                                 threshold=threshold)
# iterate through each row of scripture phrases dataset and run TFIDF model and cosine similarity.
match_extractor.run_extractor(path_matches=path_matches, git_push = True, quarto_publish=False)

match_extractor.matches_total

#%%
# git add .;git commit -m 'changes';git push;
# import pandas as pd
# cool_dict = {
#     'index_woodruff':[1,2,3,10,20],
#     'index_scriptures':[1,2,3,20,32],
#     'text_woodruff':['hello','my','name','poop','banana'],
# }

# data = pd.DataFrame(cool_dict)
# data

# #%%

# data.sort_values(['index_woodruff', 'index_scriptures'], inplace=True)
# # Create a mask to identify rows where the indices are not 1 apart
# mask = (data['index_woodruff'].diff() != 1) | (data['index_scriptures'].diff() != 1)
# mask
# data['group'] = mask.cumsum()
# data

# #%%
# # Create a new column to identify groups based on the mask
# data = data.groupby('group').agg({
#     'index_woodruff': 'last',
#     'index_scriptures': 'last',
#     # 'match_count' : 'sum',
#     # 'cosine_score': 'mean',
#     # 'verse_title': 'first',
#     # 'volume_title': 'first',
#     # 'internal_id': 'first',
#     # 'parent_id': 'first',
#     # 'order': 'first',
#     # 'website_url': 'first',
#     'text_woodruff': ' '.join,
#     # 'text_scriptures': ' '.join,
#     # 'dates': 'first',
# })
# # data['cosine_score'] = data['cosine_score'].apply(lambda x: round(x, 5))
# data

# #%%

# data = data.append({'index_woodruff':4, 'index_scriptures':4,'text_woodruff':'is porter'}, ignore_index=True)
# data

# # %%
# data.sort_values(['index_woodruff', 'index_scriptures'], inplace=True)
# # Create a mask to identify rows where the indices are not 1 apart
# mask = (data['index_woodruff'].diff() != 1) | (data['index_scriptures'].diff() != 1)
# mask
# data['group'] = mask.cumsum()
# data

# data.groupby('group').agg({
#     'index_woodruff': 'last',
#     'index_scriptures': 'last',
#     # 'match_count' : 'sum',
#     # 'cosine_score': 'mean',
#     # 'verse_title': 'first',
#     # 'volume_title': 'first',
#     # 'internal_id': 'first',
#     # 'parent_id': 'first',
#     # 'order': 'first',
#     # 'website_url': 'first',
#     'text_woodruff': ' '.join,
#     # 'text_scriptures': ' '.join,
#     # 'dates': 'first',
# })

#%%

# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
# from IPython.display import display

# documentA = 'the man went out for a walk'
# documentB = 'the children sat around the fire'
# corpus = [documentA, documentB]
# bagOfWordsA = documentA.split(' ')
# bagOfWordsB = documentB.split(' ')
# bagOfWordsA
# bagOfWordsB

# #%%
# uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))
# uniqueWords
# #%%
# print('----------- compare word count -------------------')
# numOfWordsA = dict.fromkeys(uniqueWords, 0)
# for word in bagOfWordsA:
#     numOfWordsA[word] += 1
# numOfWordsB = dict.fromkeys(uniqueWords, 0)
# for word in bagOfWordsB:
#     numOfWordsB[word] += 1
# numOfWordsB

# #%%
# series_A = pd.Series(numOfWordsA)
# series_B = pd.Series(numOfWordsB)
# df = pd.concat([series_A, series_B], axis=1).T
# df = df.reindex(sorted(df.columns), axis=1)
# display(df)

# #%%
# tf_df = df.divide(df.sum(1),axis='index')
# tf_df

# #%%

# n_d = 1+ tf_df.shape[0]
# df_d_t = 1 + (tf_df.values>0).sum(0)
# idf = np.log(n_d/df_d_t) + 1
# idf

# #%%
# pd.DataFrame(df.values * idf,
#                   columns=df.columns )



# # %%
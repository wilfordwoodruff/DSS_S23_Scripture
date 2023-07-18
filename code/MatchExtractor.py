#%%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from StringUtil import StringUtil
from tqdm import tqdm
import itertools
import pandas as pd
import subprocess



class MatchExtractor:
    """ match extractor class, pass initialize with 2 pandas dataframes and the phrase to split each row of text into
    """

    def __init__(self, data_woodruff, data_scriptures, phrase_length, threshold):
        self.matches_total = pd.DataFrame()
        self.matches_current = pd.DataFrame()
        self.phrase_length = phrase_length
        self.threshold = threshold
        # local paths
        self.__load_woodruff_data(data_woodruff)
        self.__load_scripture_data(data_scriptures)
        self.__load_vectorizer()

    def __load_woodruff_data(self, data_woodruff):
        """ save self.data_woodruff as pandas dataframe
        """
        self.data_woodruff_full = data_woodruff.copy()
        # split each journal entry into a list of phrases then explode it all
        self.data_woodruff = StringUtil.expand_dataframe_of_text(data_woodruff, 'text_woodruff', self.phrase_length)

    def __load_scripture_data(self, data_scriptures):
        """ save self.data_scripture as pandas dataframe
        """
        self.data_scriptures_full = data_scriptures.copy()
        # split each verse into a list of phrases then explode it all
        self.data_scriptures = StringUtil.expand_dataframe_of_text(data_scriptures, 'text_scriptures', self.phrase_length)

    def __load_vectorizer(self):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix_woodruff = self.vectorizer.fit_transform(self.data_woodruff['text_woodruff'])

    def run_extractor(self, path_matches, git_push = False, quarto_publish = False):
        """ Uses already trained TFIDF model, first extraction algorithm
            loops through each row of expanded scriptures dataframe and computes the tfidf vector of each scriptures phrase
            then compute the vectors of each woodruff phrase and create a vector
            then compute cosine similarity between each vector and filter by a certain threshold.
            then append everything back into a dataset.
        """
        self.data_woodruff_copy = self.data_woodruff.copy()
        progress_bar = tqdm(total=len(self.data_scriptures))
        # iterate through each row of data_scriptures_pandas dataframe and run TFIDF vectorizer on the scripture text
        for index, row_scriptures in self.data_scriptures.iterrows():
            progress_bar.update(1)
            description = f"{row_scriptures['verse_title']} total match count: {len(self.matches_total)}"
            progress_bar.set_description(description)
            # compute cosine similarity scores
            tfidf_matrix_scriptures = self.vectorizer.transform([row_scriptures['text_scriptures']])
            # compute cosine similarity scores for given verse for each phrase in woodruff dataset
            cosine_scores = cosine_similarity(self.tfidf_matrix_woodruff, tfidf_matrix_scriptures)
            self.data_woodruff_copy['cosine_score'] = cosine_scores.flatten()
            self.data_woodruff_copy['text_scriptures'] = row_scriptures['text_scriptures']
            self.matches_current = self.data_woodruff_copy.rename_axis('index_woodruff').reset_index()
            self.matches_current['index_scriptures'] = index
            self.matches_current['verse_title']  = row_scriptures['verse_title']
            self.matches_current['volume_title'] = row_scriptures['volume_title']
            self.matches_current['book_title']   = row_scriptures['book_title']
            # filter matches by threshold
            self.matches_current = self.matches_current.query("cosine_score > @self.threshold")
            self.matches_total = pd.concat([self.matches_total, self.matches_current])
            self.matches_total = self.matches_total#.sort_values(by='cosine_score', ascending=False)

            # save to file
            self.resolve_extensions()
            self.matches_total.to_csv(path_matches, index=False)
        self.matches_total.to_csv(path_matches, index=False)

        progress_bar.close()

        if git_push:
            self.git_push()

        if quarto_publish:
            self.quarto_publish()

    def resolve_extensions(self):
        self.matches_total.sort_values(['index_woodruff', 'index_scriptures'], inplace=True)
        # Create a mask to identify rows where the indices are not 1 apart
        mask = (self.matches_total['index_woodruff'].diff() != 1) | (self.matches_total['index_scriptures'].diff() != 1)
        # Create a new column to identify groups based on the mask
        self.matches_total['group'] = mask.cumsum()
        self.matches_total['match_count'] = 1
        self.matches_total = self.matches_total.groupby('group').agg({
            'index_woodruff': 'last',
            'index_scriptures': 'last',
            'cosine_score': 'mean',
            'verse_title': 'first',
            'volume_title': 'first',
            'internal_id': 'first',
            'parent_id': 'first',
            'order': 'first',
            'website_url': 'first',
            'match_count' : 'count',
            'text_woodruff': ' '.join,
            'text_scriptures': ' '.join,
            'dates': 'first',
        })
        self.matches_total['cosine_score'] = self.matches_total['cosine_score'].apply(lambda x: round(x, 5))

    @staticmethod
    def git_push():
        commands = ['git add .','git commit -m "new matches"','git pull','git push']

        subprocess.run(commands[0], shell = True, encoding = 'utf-8')
        subprocess.run(commands[1], shell = True, encoding = 'utf-8')
        subprocess.run(commands[2], shell = True, encoding = 'utf-8')
        subprocess.run(commands[3], shell = True, encoding = 'utf-8')

    @staticmethod
    def quarto_publish():
        command = 'quarto publish'
        subprocess.run(command, shell = True, input = 'y\n', encoding = 'utf-8')
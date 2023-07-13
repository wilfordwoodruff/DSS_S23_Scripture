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

    def __init__(self, data_woodruff, data_scriptures, phrase_length, threshold, path_root):
        self.matches_total = pd.DataFrame()
        self.matches_current = pd.DataFrame()
        self.phrase_length = phrase_length
        self.threshold = threshold
        self.path_matches             = path_root + 'data_matches.csv'
        self.path_matches_temporary   = path_root + 'data_matches_temporary.csv'
        self.path_matches_extensions = path_root + 'data_matches_extensions.csv'
        self.path_matches_extensions_temporary = path_root + 'data_matches_extensions_temporary.csv'
        # local paths
        self.__load_woodruff_data(data_woodruff)
        self.__load_scripture_data(data_scriptures)
        self.__load_vectorizer()

    def __load_woodruff_data(self, data_woodruff):
        """ save self.data_woodruff as pandas dataframe
        """
        self.data_woodruff_full = data_woodruff.copy()
        # split each journal entry into a list of phrases then explode it all
        self.data_woodruff = StringUtil.expand_dataframe_of_text(data_woodruff, 'text', self.phrase_length)

    def __load_scripture_data(self, data_scriptures):
        """ save self.data_scripture as pandas dataframe
        """
        self.data_scriptures_full = data_scriptures.copy()
        # split each verse into a list of phrases then explode it all
        self.data_scriptures = StringUtil.expand_dataframe_of_text(data_scriptures, 'text', self.phrase_length)

    def __load_vectorizer(self):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix_woodruff = self.vectorizer.fit_transform(self.data_woodruff['text'])

    def run_extractor(self, extensions = True, save = False, quarto_publish = False):
        """ Uses already trained TFIDF model, first extraction algorithm
            loops through each row of expanded scriptures dataframe and computes the tfidf vector of each scriptures phrase
            then compute the vectors of each woodruff phrase and create a vector
            then compute cosine similarity between each vector and filter by a certain threshold.
            then append everything back into a dataset.
        """
        self.progress_bar = tqdm(total=len(self.data_scriptures))
        # iterate through each row of data_scriptures_pandas dataframe and run TFIDF vectorizer on the scripture text
        for index, row_scriptures in self.data_scriptures.iterrows():
            self.progress_bar.update(1)
            description = f"{row_scriptures['verse_title']} total match count: {len(self.matches_total)}"
            self.progress_bar.set_description(description)
            # compute cosine similarity scores for given verse
            self.matches_current = self.extract_tfidf_percentage_matches(row_scriptures['text'])
            self.matches_current['verse_title']  = row_scriptures['verse_title']
            self.matches_current['volume_title'] = row_scriptures['volume_title']
            self.matches_current['book_title']   = row_scriptures['book_title']


            # filter matches by threshold
            self.matches_current = self.matches_current.query("cosine_score > @self.threshold")
            if len(self.matches_current) > 0:
                self.matches_total = pd.concat([self.matches_total, self.matches_current]).sort_values(
                    by='cosine_score', ascending=False)[['date', 'verse_title', 'cosine_score', 'phrase_woodruff','phrase_scripture', 'volume_title']]

                # save to file
                self.matches_total.to_csv(self.path_matches_temporary, index=False)

        self.progress_bar.close()

        if extensions:
            self.filter_to_matches_only_data()
            self.extract_matches_extensions()

        if save:
            self.matches_total.to_csv(self.path_matches, index=False)
            self.matches_extensions.to_csv(self.path_matches_extensions, index = False)

        if quarto_publish:
            self.quarto_publish()

    def filter_to_matches_only_data(self):
        """ The first extraction algorithm is a much smaller runtime so it is run first.
            Then second algorithm is run that also finds extensions which takes longer to run.
            This method filters the data to only verses and entries that matches have already been found within.
        """
        # filter input data down to rows that already contain matches
        if not len(self.matches_total) > 0:
            self.matches_total =  pd.read_csv(self.path_matches_temporary)
        date_matches = list(self.matches_total['date'].unique())
        self.data_woodruff_filtered = self.data_woodruff.query('date in @date_matches')[['date','text']]

        verse_matches = list(self.matches_total['verse_title'].unique())
        self.data_scriptures_filtered = self.data_scriptures.query('verse_title in @verse_matches')

    def extract_tfidf_percentage_matches(self, scripture_text):
        """ Pass in a single string and it returns a pandas dataframe containing the woodruff phrases along with the cosine similarity value
        """
        tfidf_matrix_scriptures = self.vectorizer.transform([scripture_text])
        cosine_scores = cosine_similarity(self.tfidf_matrix_woodruff, tfidf_matrix_scriptures)
        cosine_scores = pd.DataFrame(cosine_scores, columns=['cosine_score']).apply(lambda x: round(x, 5))
        cosine_scores['phrase_woodruff'] = list(self.data_woodruff['text'])
        cosine_scores['date'] = list(self.data_woodruff['date'])
        cosine_scores['phrase_scripture'] = scripture_text
        return cosine_scores

    def extract_matches_extensions(self):
        """ Double for loop looping through expanded woodruff entries dataframe and expanded scriptures dataframe
            checks each phrase with each other phrase. If the score is above the threshold it uses a while loop to check for extensions of each phrase
            index values are stored in a list each time a score is calculated for each phrase
            that way it can detect if 2 phrases have already been compared and skip to the next phrases
            it appends all matche data to lists and creates a pandas dataframe with each list as a column to be saved to a csv
        """
        matches_woodruff = []
        matches_scriptures = []
        scores = []
        total_match_indices = []
        dates = []
        verse_titles = []
        volume_titles = []
        progress_bar = tqdm(total=len(list(self.data_woodruff_filtered['text'])) - 1)
        extension_count = 0
        for index1, text_woodruff in enumerate(list(self.data_woodruff_filtered['text'])):
            progress_bar.update(1)
            current_date = self.data_woodruff_filtered.iloc[index1]['date']
            progress_bar.set_description('current_date:'+str(current_date)+'extensions found: ' + str(extension_count))
            for index2, text_scriptures in enumerate(list(self.data_scriptures_filtered['text'])):
                current_match_indices = []
                if (index1, index2) in list(itertools.chain.from_iterable(total_match_indices)):
                    # print('repeat:', (index1, index2))
                    continue
                current_match_indices.append((index1, index2))
                text_woodruff_copy = text_woodruff
                text_scriptures_copy = text_scriptures
                score = StringUtil.compute_similarity(self.vectorizer, text_woodruff_copy, text_scriptures_copy)
                if score > self.threshold:
                    current_verse_title = self.data_scriptures_filtered.iloc[index2]['verse_title']
                    current_volume_title = self.data_scriptures_filtered.iloc[index2]['volume_title']
                    index1_extension = index1
                    index2_extension = index2
                    while True:
                        index1_extension += 1
                        index2_extension += 1
                        if index1_extension > len(list(self.data_woodruff_filtered['text']))-1:
                            break
                        if index2_extension > len(list(self.data_scriptures_filtered['text']))-1:
                            break
                        text_woodruff_extension = list(self.data_woodruff_filtered['text'])[index1_extension]
                        text_scriptures_extension = list(self.data_scriptures_filtered['text'])[index2_extension]
                        score_extension = StringUtil.compute_similarity(self.vectorizer, text_woodruff_extension, text_scriptures_extension)
                        if score_extension > self.threshold:
                            extension_count += 1
                            current_match_indices.append((index1_extension, index2_extension))
                            # print('adding extensions...', (index1_extension, index2_extension))
                            text_woodruff_copy += ' ' + text_woodruff_extension
                            text_scriptures_copy += ' ' + text_scriptures_extension
                        else:
                            break
                    # compute new score with extensions included
                    score = StringUtil.compute_similarity(self.vectorizer, text_woodruff_copy, text_scriptures_copy)
                    total_match_indices.append(current_match_indices)
                    matches_woodruff.append(text_woodruff_copy)
                    matches_scriptures.append(text_scriptures_copy)
                    dates.append(current_date)
                    verse_titles.append(current_verse_title)
                    volume_titles.append(current_volume_title)
                    scores.append(score)
                    matches_dict = {
                        'date' : dates,
                        'verse_title' : verse_titles,
                        'volume_title' : volume_titles,
                        'phrase_indices' : total_match_indices,
                        'score' : scores,
                        'matches_woodruff' : matches_woodruff,
                        'matches_scriptures' : matches_scriptures,
                    }
                    self.matches_extensions = pd.DataFrame(matches_dict).sort_values(by = 'score', ascending=False)
                    self.matches_extensions.to_csv(self.path_matches_extensions_temporary, index = False)
        progress_bar.close()

    @staticmethod
    def quarto_publish():
        command = 'quarto publish'
        subprocess.run(command, shell = True, input = 'y\n', encoding = 'utf-8')

#%%

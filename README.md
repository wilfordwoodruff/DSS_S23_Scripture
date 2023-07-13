# DSS_S23_Scripture
Society's Scripture Matching


## code
- [run_extractor.py](code/run_extractor.py) python script that cleans the data and runs the MatchExtractor class comparing the scripture dataset phrases with the woodruff journal entries dataset.

- [MatchExtractor.py](code/MatchExtractor.py) class that takes in Woodruff journal entries dataset, splits the entries into a list of `phrase_length` word phrases, then loops through all entries and scriptures to find matches. includes extensions so that matches over `phrase_langth` can be found.

- [StringUtil.py](code/StringUtil.py) string and pandas dataframe utility class to fix some the existing pandas dataframe methods.

## data
- [data_matches_extensions.csv](data/data_matches_extensions.csv) csv containing all the current matches found.
- [data_woodruff_raw.csv](data/data_woodruff_raw.csv) csv containing original woodruff entries. 
- [data_scriptures.csv](data/data_scriptures.csv) clean csv of all verses in standard works.

## in progress
- add internal id, parent id, order columns to matches
- streamlit app with some analysis
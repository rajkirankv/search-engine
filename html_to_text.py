import logging
import os
import sys
import time

import bs4
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# To do
# Extract relevant metadata and save it. Assign weights to each word.
# For each word and document pair, calculate necessary numbers. Implement search queries
# Display results


DATA_PATH = '/Users/stan/Dropbox/UCI/Academics/Winter17/CS221/projects/project3/data/WEBPAGES_RAW/'
# DATA_PATH = '/Users/stan/Dropbox/UCI/Academics/Winter17/CS221/projects/project3/data/test'
BATCH_SIZE = 5
TRACKING_SIZE = 1
TIME_FACTOR = 60
LANGUAGE = 'english'
logging.basicConfig(stream=sys.stderr, level=logging.INFO)


def file_isvalid(filename, filepath):
    # Checks: 1. File starts with a '.'
    # 2. File extension other than html
    # Open the file, use beautiful soup to check if it is indeed html
    if filename.startswith('.') or filename.endswith('.json') or filename.endswith('.tsv'):
        return False

    file_abs_path = os.path.join(filepath, filename)

    soup = bs4.BeautifulSoup(open(file_abs_path), 'lxml')
    if not bool(soup.find()):
        return False

    return True


def clean_text(dirty_text):
    # Accept a string, tokenize, remove stop words, stem the remaining text and return a clean string
    regex_pattern = r"[\w']+"
    tokenizer = RegexpTokenizer(regex_pattern)
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    tokens = tokenizer.tokenize(dirty_text)
    stop_word_set = set(stopwords.words(LANGUAGE))
    clean_list = [token for token in tokens if token not in stop_word_set]

    for i in range(len(clean_list)):
        try:
            clean_list[i] = stemmer.stem(clean_list[i])
        except IndexError as err:
            continue

    # clean_list = [lemmatizer.lemmatize(token) for token in clean_list]
    clean_string = ' '.join(clean_list)

    return clean_string


def read_files(list_of_files):
    # Given a list of file paths containing html files, return a dictionary as file:string pairs
    file_strings = dict()  # This is a BIG dictionary with file: wordlist pairs
    files_read = 0
    for file_path in list_of_files:
        soup = bs4.BeautifulSoup(open(file_path), 'lxml')
        soup_string = soup.getText()
        clean_string = clean_text(soup_string)
        file_strings[file_path] = clean_string
        files_read += 1
        if files_read % TRACKING_SIZE == 0:
            logging.debug("HTML files processed: {}".format(files_read))

    return file_strings


def process_batch(files_list, tf_idf_matrix):
    file_strings = read_files(files_list)
    tf_idf_matrix.fit(file_strings.values())


def get_tf_idf(corpus=DATA_PATH):
    files_list = list()
    files_processed = 0
    # file_strings = dict()
    tf_idf_matrix = TfidfVectorizer()

    logging.info('Calculating tf-idf')
    for root, dirnames, filenames in os.walk(corpus):
        for filename in filenames:
            if file_isvalid(filename=filename, filepath=root):
                file_abs_path = os.path.join(root, filename)
                files_list.append(file_abs_path)
                files_processed += 1
                if files_processed % TRACKING_SIZE == 0:
                    logging.debug('File paths processed: {}'.format(files_processed))
                if files_processed >= BATCH_SIZE:
                    logging.debug('Batches finished: {}, Batch size: {}'.format(files_processed // BATCH_SIZE, BATCH_SIZE))
                    process_batch(files_list=files_list, tf_idf_matrix=tf_idf_matrix)
                    files_processed = 0
                    files_list = list()
    process_batch(files_list=files_list, tf_idf_matrix=tf_idf_matrix)
    return tf_idf_matrix


def get_results(query, tf_idf_matrix):
    score = tf_idf_matrix.transform([query])
    return score


def main(query='linux ics computer', build=True):
    if build is True:
        tf_idf_matrix = get_tf_idf(DATA_PATH)
    clean_query = clean_text(query)
    results = get_results(clean_query, tf_idf_matrix)
    feature_names = tf_idf_matrix.get_feature_names()
    for col in results.nonzero()[1]:
        print(feature_names[col], ' - ', results[0, col])


if __name__ == '__main__':
    t1 = time.clock()
    main()
    t2 = time.clock()
    time_taken = t2 - t1
    print("Time taken: {} minutes {} seconds".format(time_taken // TIME_FACTOR, time_taken % TIME_FACTOR))

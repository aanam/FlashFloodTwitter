from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

def main(in_file, out_folder, save):
    # open input file with all tweets separated by a new_line
    text_file = open(in_file)

    # read all the tweets, eliminate numbers and white sapces and store them in a list where each element is a tweet
    tweets = (text_file.read().strip()).translate(None, '1234567890').split('\n')

    # define a dictionary of stop words
    stopset = set(stopwords.words('english'))
    freq_words = ["ellicott", "city", "ellicottcity", "md", "maryland", "in", "elliott"]
    stopset |= set(freq_words)



# list for tokenized documents in loop
texts = []


# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary, passes=20)
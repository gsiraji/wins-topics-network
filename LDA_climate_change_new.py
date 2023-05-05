# from https://medium.com/analytics-vidhya/topic-modeling-using-gensim-lda-in-python-48eaa2344920

import nltk
nltk.download('stopwords')
import re
import numpy as np
import pandas as  pd
from pprint import pprint# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel# spaCy for preprocessing
import spacy# Plotting tools
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
# %matplotlib inline
import parquet
import os

# Prepare stopwords
# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# load daaset
path = os.getcwd()
df = pd.read_parquet('{}/project_data/arxiv_climate_change.parquet'.format(path), engine='auto')

for col in df.columns:
    print(col)

# Remove newline characters
# Convert to list 
data = df.abstract.values.tolist()  
# Remove new line characters 
data = [re.sub('\s+', ' ', sent) for sent in data]  
# Remove distracting single quotes 
data = [re.sub("\'", "", sent) for sent in data]  
pprint(data[:1])

# Tokenize words and cleanup the text
def sent_to_words(sentences):
  for sentence in sentences:
    yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))            
    #deacc=True removes punctuations

data_words = list(sent_to_words(data))
print(data_words[:1])

# Remove Stopwords, make bigrams and lemmatize
# Define function for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    # Build the bigram
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100) # higher threshold fewer phrases.

    # Faster way to get a sentence clubbed as a bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    # Build the trigram models
    bigram = gensim.models.Phrases(texts, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[texts], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # See trigram example
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Call preprocessing functions in order
# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
# nlp = spacy.load('en', disable=['parser', 'ner'])
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])


# Create Dictionary and Corpus needed for Topic Modeling
# Create Dictionary 
id2word = corpora.Dictionary(data_lemmatized)  
# Create Corpus 
texts = data_lemmatized  
# Term Document Frequency 
corpus = [id2word.doc2bow(text) for text in texts]  
# View 
print(corpus[:1])

# Building topic model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=15, 
                                           random_state=100,
                                           update_every=1, # determines how often the model parameters should be updated
                                           chunksize=100, # the number of documents to be used in each training chunk
                                           passes=10, # the total number of training passes
                                           alpha='auto',
                                           per_word_topics=True)


# Print the keyword of topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]




# Evaluate topic models

# Compute model Perplexity and Coherence score
# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  
# a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
#Higher the topic coherence, the topic is more human interpretable.


# Visualize the topic model
# Visualize the topics
# pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(vis, "{}/LDA/LDA_visual.html".format(path)) 


# get value of topic per document
# got this from: https://notebook.community/gojomo/gensim/docs/notebooks/topic_methods
all_topics = lda_model.get_document_topics(corpus, per_word_topics=True)

doc_list = []
for doc_topics, word_topics, phi_values in all_topics:
   doc_list.append(doc_topics)

# create documents x topic df
df = pd.DataFrame(doc_list)

df_all = df[0]
df_all = df_all.reset_index()
df_all = df_all.rename({'index': 'document', 0: 'tw' }, axis=1)

for val in df.columns:
    dval = df[val]
    dval = dval.reset_index()
    dval = dval.rename({'index': 'document', val: 'tw' }, axis=1)
    df_all = pd.concat([df_all,dval])

df_all[['topic', 'weight']] = pd.DataFrame(df_all['tw'].tolist(), index=df_all.index)
df_all.drop('tw', axis = 1)
df_all = df_all.dropna()
df_all['topic'] = df_all['topic'].astype(int)

df_doc_topic = df_all.pivot_table(index='document', 
                        columns='topic', 
                        values='weight')

# rename topic columns T#
for val in df.columns:
    df_doc_topic = df_doc_topic.rename({val: 'T{}'.format(val) }, axis=1)

df_doc_topic = df_all.pivot_table(index='document', 
                        columns='topic', 
                        values='weight')

# save
df_doc_topic.to_csv("{}/project_data/document_topic.csv".format(path))


# create df with top 3 words for each topic
x = lda_model.show_topics(num_topics=15, num_words=3,formatted=False)
topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]
print(topics_words)
df_words = pd.DataFrame(topics_words)
df_words = df_words.rename({0: 'Topic', 1: 'words' }, axis=1)

# change all values > 0 to 1
df_doc_topic_1s = df_doc_topic
for col in df_doc_topic_1s.columns:
   df_doc_topic_1s.loc[df_doc_topic_1s[col] > 0, col] = 1

# create topic freq df
df_topic_freq = pd.DataFrame()
for val1 in df_doc_topic_1s.columns:
    df_sub = df_doc_topic.groupby([val1]).size().reset_index(name="frequency")
    df_sub["Topic"] = val1
    df_sub = df_sub.drop([val1], axis = 1)

    df_topic_freq = pd.concat([df_sub,df_topic_freq])

# add top 3 words
df_topic_freq = df_topic_freq.merge(df_words, how='left', on='Topic')

# save
df_topic_freq.to_csv("{}/project_data/topic_freq.csv".format(path))

# create topic x topic freq df
df_topic = pd.DataFrame()
for val1 in df_doc_topic_1s.columns:
    for val2 in df_doc_topic_1s.columns:
        if val1 == val2:
            x = 1
        else:
            df_sub = df_doc_topic.groupby([val1, val2]).size().reset_index(name="frequency")
            df_sub["TopicA"] = val1
            df_sub["TopicB"] = val2
            df_sub = df_sub.drop([val1,val2], axis = 1)

            df_topic = pd.concat([df_sub,df_topic])

# add top 3 words
df_wordsA = df_words.rename({"Topic": "TopicA", "words": "TopicA_words"}, axis=1)
df_topic = df_topic.merge(df_wordsA, how='left', on='TopicA')
df_wordsB = df_words.rename({"Topic": "TopicB", "words": "TopicB_words"}, axis=1)
df_topic = df_topic.merge(df_wordsB, how='left', on='TopicB')

# save
df_topic.to_csv("{}/project_data/topic_topic_freq.csv".format(path))

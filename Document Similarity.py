import concurrent.futures
from datetime import datetime
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, WordEmbeddingSimilarityIndex, Word2Vec, LdaModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.keyedvectors import KeyedVectors
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity
from gensim.similarities import MatrixSimilarity, Similarity
from itertools import combinations
import joblib
from multiprocessing import Manager
import nltk
from nltk.corpus import stopwords
import numpy as np
from os import listdir
import pandas as pd
#!pip install rank-bm25
from rank_bm25 import BM25Plus
import re
import sqlalchemy as sal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, cosine_distances
from string import punctuation
from tqdm import tqdm

def getSentences(text):
    """
    Split the text into sentences by dots provided:
    It is not a dot among the list of common abbreviations such as art., arts., s.a., n.º, doc., docs., fl., fls.,
    dr., drs., cia., cias.
    """

    regex = r'(?<!art)(?<!arts)(?<!\.)(?<!\()(?<!\s[asn])(?<!s.a)(?<!doc)(?<!docs)(?<!fls)(?<!fl)(?<!dr)(?<!drs)(?<!cia)(?<!cias)([\.]{1}\B)'
    if len(text) > 1:
        return [word for word in re.split(regex, text) if len(word) > 1]
    else:
        return ''

def regex_substitute(document, regex_list):
    for regex in regex_list:
        document = re.sub(regex, '', document)
    return document

def tagging(text, tags_to_use):
    tagged = brill_tagger.tag(nltk.word_tokenize(text))
    return ' '.join([word for word, tag in tagged if tag in tags_to_use])

def tokenizer_stemmer(text):
    token_list = nltk.word_tokenize(text, language='portuguese')
    token_list = [token for token in token_list if re.search(r"(?u)\b[0-9a-zÀ-ÿ-]{3,}\b", token)]
    token_list = [token for token in token_list if token not in stoplist]
    token_list = [stemmer.stem(token) for token in token_list]
    listToStr = ' '.join([str(elem) for elem in token_list])
    
    return listToStr

def tokenizer(text):
    token_list = nltk.word_tokenize(text, language='portuguese')
    token_list = [token for token in token_list if re.search(r"(?u)\b[0-9a-zÀ-ÿ-]{3,}\b", token)]
    token_list = [token for token in token_list if token not in stoplist]
    listToStr = ' '.join([str(elem) for elem in token_list])
    
    return listToStr

def topNSimilarViolations(sim_matrix, n, violationId):
    """
    Returns the 'n' most similar violations for a 'violationId'
    'sim_matrix' is a similarity dataframe
    """
    
    sim_series = sim_matrix[(sim_matrix.index != violationId)][violationId]
    sim_series = sim_series[sim_series < 0.999].nlargest(n)
    
    return sim_series

def cosineSimilarityByViolation(tfidfMatrix):
    
    startrow = 0 
    level2index = data_orig_sent_.index.get_level_values(1)
    violations = np.unique(level2index)
    fullSimDf = pd.DataFrame(index=violations)

    for violation in tqdm(violations):
        nrows = data_orig_sent_.loc[(slice(None), [violation]), :].shape[0]
        endrow = startrow + nrows
        sim = cosine_similarity(tfidfMatrix[startrow:endrow,:], tfidfMatrix)
        simDf = pd.DataFrame(data=sim,
                             index=np.full(nrows, violation),
                             columns=level2index)
        t = simDf.groupby(level=0).mean()
        z = t.T.groupby(level=0).mean()
        fullSimDf = fullSimDf.join(z)
        startrow = startrow + nrows
    
    return fullSimDf

#engine = sal.create_engine('mssql+pyodbc://LAPTOP-NNDGMEMB/beth?driver=ODBC+Driver+13+for+SQL+Server?Trusted_Connection=yes')
#conn = engine.connect()
#data = pd.read_sql_table('violations_content', conn)
data = pd.read_csv('./data.csv')

# Remove the rows in which the document content is null.
# It happens when documents (documentoId) iniatilly loaded into 'violations_content'
# can't have their content retrieved (e.g. PDF that can't be parsed by PDFMiner)
data = data[~data['docContSplitted'].isna()]

# possible 'serieId' for documents that represent the initial document in a violation process

#serieId  serieNome                     
#344      REPRESENTAÇÃO - Eletrônica        887
#42       DEFESA                            405
#381      RECLAMAÇÃO - Atendimento SUSEP    151
#173      REPRESENTAÇÃO                      49
#341      AUTO - Infração Eletrônico         18
#61       RECLAMAÇÃO                         12
#45       DENÚNCIA                            1

initial_docs = [173,  344, 341, 381,  61,  45]

# filter the dataframe to keep only one document per violation and remove 'DEFESA'

data_orig = data.loc[data['serieId'].isin(initial_docs),:]
data_orig = data_orig.sort_values(by='documentoId', ascending=False)
data_orig = data_orig.drop_duplicates(subset='infracaoId', keep='first')

columns_to_keep = ['infracaoId', 'processoId', 'docContSplitted']

data_orig = data_orig[columns_to_keep]

data_orig.reset_index(drop=True, inplace=True)

data_orig['docContSentences'] = data_orig['docContSplitted'].apply(getSentences)

data_orig_sent = data_orig.explode('docContSentences')

# list of regex to remove expressions which don't carry information
# 0: 'coordenação (geral) de'
# 1: 'ministério da fazenda|economia'
# 2: 'superintendência de seguros privados'
# 3: brazilian postal code
# 4: URL
# 5: brazilian phone numbers
# 6: brazilian addresses
# 7: e-mail (emailregex.com)

regex_list = [r'(coordena[cç][aã]o)\s?-?(geral)?\s?de', r'(minist[eé]rio)\s(da)\s(fazenda|economia)',
 r'(superintend[eê]ncia)\sde\sseguros\sprivados', r'(cep)\:?\s*\b\d{2}\.?\d{3}-?\d{3}\b',
r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})',
r'\b(\+55\s?)?\(?([0-9]{2,3}|0((x|[0-9]){2,3}[0-9]{2}))\)?\s*[0-9]{4,5}[- ]*[0-9]{4}\b',
r'\b(rua|r\.|avenida|av\.?|travessa|trav\.?|largo|quadra|qd|alameda|conjunto|conj\.?|estrada|pra[cç]a|rodovia|rod\.?)\s([a-zA-Z_\s]+)[, ]+(\d+)\s?([-/\da-zDA-Z\\ ]+)?\b',
r"(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])",
]

data_orig['regexCleanedText'] = data_orig['docContSplitted'].apply(regex_substitute, args=(regex_list,))

data_orig_sent['regexCleanedText'] = data_orig_sent['docContSentences'].apply(regex_substitute, args=(regex_list,))

# PoS Tagger
# Load trained POS tagger

brill_tagger = joblib.load('./POS_tagger_brill.pkl')

concepts = ['N', 'N|EST']
conceptsAndRelations = ['N', 'N|EST', 'V', 'V|!', 'V|+', 'V|EST']

data_orig['concepts'] = data_orig['regexCleanedText'].apply(tagging, args=(concepts,))
data_orig['conceptsAndRelations'] = data_orig['regexCleanedText'].apply(tagging, args=(conceptsAndRelations,))

stemmer = nltk.stem.RSLPStemmer()
stoplist = set(stopwords.words('portuguese') + list(punctuation))

data_orig['stemTextFull'] = data_orig['regexCleanedText'].apply(tokenizer_stemmer)
data_orig['stemTextConcepts'] = data_orig['concepts'].apply(tokenizer_stemmer)
data_orig['stemTextConceptsAndRelations'] = data_orig['conceptsAndRelations'].apply(tokenizer_stemmer)
data_orig_sent['stemSentFull'] = data_orig_sent['regexCleanedText'].apply(tokenizer_stemmer)

#### TF-IDF vectorization and Cosine Similarities ####

#vectorizer
vect = TfidfVectorizer(strip_accents=None,
                       lowercase=False,
                       max_df=0.8,
                       min_df=1,
                       tokenizer=None,
                       analyzer='word',
                       ngram_range=(2,3),
                       vocabulary=None)

textFull_tfidf = vect.fit_transform(data_orig['stemTextFull'])
textConcepts_tfidf = vect.fit_transform(data_orig['stemTextConcepts'])
textConceptsAndRelations_tfidf = vect.fit_transform(data_orig['stemTextConceptsAndRelations'])

data_orig_sent_ = data_orig_sent.sort_values(by=['infracaoId']).reset_index(drop=True)
data_orig_sent_ = data_orig_sent_.set_index([data_orig_sent_.index, 'infracaoId'])

sentFull_tfidf = vect.fit_transform(data_orig_sent_['stemSentFull'])

textFull_cossim = pd.DataFrame(data=cosine_similarity(textFull_tfidf),
                               index=data_orig['infracaoId'].tolist(),
                               columns=data_orig['infracaoId'].tolist())

textFull_cossim.to_csv("./results/tfidfTextFullCossim.csv")

textConcepts_cossim = pd.DataFrame(data=cosine_similarity(textConcepts_tfidf),
                               index=data_orig['infracaoId'].tolist(),
                               columns=data_orig['infracaoId'].tolist())

textConcepts_cossim.to_csv("./results/tfidfTextConceptsCossim.csv")

textConceptsAndRelations_cossim = pd.DataFrame(data=cosine_similarity(textConceptsAndRelations_tfidf),
                               index=data_orig['infracaoId'].tolist(),
                               columns=data_orig['infracaoId'].tolist())

textConceptsAndRelations_cossim.to_csv("./results/tfidfTextConceptsAndRelationsCossim.csv")

sentFull_cossim = cosineSimilarityByViolation(sentFull_tfidf)

sentFull_cossim.to_csv("./results/tfidfSentFullCossim.csv")

#### Creating the corpora for training W2V, D2V, LDA and BM25 models ####

textFullNoStem_corpus = [document.split() for document in data_orig['regexCleanedText']]

textFull_corpus = [document.split() for document in data_orig['stemTextFull']]
textConcepts_corpus = [document.split() for document in data_orig['stemTextConcepts']]
textConceptsAndRelations_corpus = [document.split() for document in data_orig['stemTextConceptsAndRelations']]
sentFull_corpus = [document.split() for document in data_orig_sent_['stemSentFull']]

#### Corpora that only includes the violations included in the gold standard ####

goldStdViolations = pd.read_excel('vioWithData.xlsx')['infracaoId'].sort_values(ascending=True).to_list()

goldTextFullNoStem_corpus = data_orig.sort_values(by=['infracaoId']).loc[data_orig['infracaoId'].isin(goldStdViolations),'regexCleanedText']
goldTextFullNoStem_corpus = [document.split() for document in goldTextFullNoStem_corpus]

goldTextFull_corpus = data_orig.sort_values(by=['infracaoId']).loc[data_orig['infracaoId'].isin(goldStdViolations),'stemTextFull']
goldTextFull_corpus = [document.split() for document in goldTextFull_corpus]

goldTextConcepts_corpus = data_orig.sort_values(by=['infracaoId']).loc[data_orig['infracaoId'].isin(goldStdViolations),'stemTextConcepts']
goldTextConcepts_corpus = [document.split() for document in goldTextConcepts_corpus]

goldTextConceptsAndRelations_corpus = data_orig.sort_values(by=['infracaoId']).loc[data_orig['infracaoId'].isin(goldStdViolations),'stemTextConceptsAndRelations']
goldTextConceptsAndRelations_corpus = [document.split() for document in goldTextConceptsAndRelations_corpus]

goldSentFull_corpus = data_orig_sent_.sort_index(level=1).loc[(slice(None), goldStdViolations), 'stemSentFull']
goldSentFull_corpus = [document.split() for document in goldSentFull_corpus]

#### Word2Vec & Doc2Vec Models and Cosine Similarities ####

# Loading a pre-trained Word2Vec model from NILC-USP (CBOW 300)
# Remember it matches full words instead of stemmed ones

#preTrainedCbow300 = KeyedVectors.load_word2vec_format('../../Code/nilc_word2vec/cbow_s300.txt', binary=False)
#preTrainedCbow300.save('preTrainedCbow300.model')
#preTrainedCbow300 = Word2Vec.load('preTrainedCbow300.model')
preTrainedCbow300_wordvectors = KeyedVectors.load('preTrainedCbow300.model')
#del preTrainedCbow300

# Training Word2Vec model from the existing complete corpus

#trainedCbowFull = Word2Vec(sentences=textFull_corpus, min_count=1, workers=4)
#trainedCbowFull.save('trainedCbowFull.model')
trainedCbowFull = Word2Vec.load('trainedCbowFull.model')
trainedCbowFull_wordvectors = trainedCbowFull.wv

# Training Word2Vec model from the existing concepts corpus

#trainedCbowConcepts = Word2Vec(sentences=textConcepts_corpus, min_count=1, workers=8)
#trainedCbowConcepts.save('trainedCbowConcepts.model')
trainedCbowConcepts = Word2Vec.load('trainedCbowConcepts.model')
trainedCbowConcepts_wordvectors = trainedCbowConcepts.wv

# Training Word2Vec model from the existing concepts and relations corpus

#trainedCbowConceptsAndRelations = Word2Vec(sentences=textConceptsAndRelations_corpus, min_count=1, workers=8)
#trainedCbowConceptsAndRelations.save('trainedCbowConceptsAndRelations.model')
trainedCbowConceptsAndRelations = Word2Vec.load('trainedCbowConceptsAndRelations.model')
trainedCbowConceptsAndRelations_wordvectors = trainedCbowConceptsAndRelations.wv

# Training Word2Vec model from the existing sentences corpus

#trainedCbowSent = Word2Vec(sentences=sentFull_corpus, min_count=1, workers=8)
#trainedCbowSent.save('trainedCbowSent.model')
trainedCbowSent = Word2Vec.load('trainedCbowSent.model')
trainedCbowSent_wordvectors = trainedCbowSent.wv

# Training a Doc2Vec model from the existing complete corpus

#documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(textFull_corpus)]
#trainedDoc2VecFull = Doc2Vec(documents, min_count=1, workers=8)
#trainedDoc2VecFull.save('trainedDoc2VecFull.model')
trainedDoc2VecFull = Doc2Vec.load('trainedDoc2VecFull.model')

# Training a Doc2Vec model from the existing concepts corpus

#documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(textConcepts_corpus)]
#trainedDoc2VecConcepts = Doc2Vec(documents, min_count=1, workers=8)
#trainedDoc2VecConcepts.save('trainedDoc2VecConcepts.model')
trainedDoc2VecConcepts = Doc2Vec.load('trainedDoc2VecConcepts.model')

# Training a Doc2Vec model from the existing concepts and relations corpus

#documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(textConceptsAndRelations_corpus)]
#trainedDoc2VecConceptsAndRelations = Doc2Vec(documents, min_count=1, workers=8)
#trainedDoc2VecConceptsAndRelations.save('trainedDoc2VecConceptsAndRelations.model')
trainedDoc2VecConceptsAndRelations = Doc2Vec.load('trainedDoc2VecConceptsAndRelations.model')

# Training a Doc2Vec model from the sentences corpus

#documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentFull_corpus)]
#trainedDoc2VecSent = Doc2Vec(documents, min_count=1, workers=4)
#trainedDoc2VecSent.save('trainedDoc2VecSent.model')
trainedDoc2VecSent = Doc2Vec.load('trainedDoc2VecSent.model')

# Creating lists of models and respective evaluation corpus

word2vecModels = [#trainedCbowFull_wordvectors,
                  #trainedCbowConcepts_wordvectors,
                  #trainedCbowConceptsAndRelations_wordvectors,
                  #preTrainedCbow300_wordvectors,
                  trainedCbowSent_wordvectors,
                 ]

word2vecCorpus = [#textFull_corpus,
                  #textConcepts_corpus,
                  #textConceptsAndRelations_corpus,
                  #textFullNoStem_corpus,
                  sentFull_corpus,
                 ]

word2vecGoldCorpus = [#goldTextFull_corpus,
                      #goldTextConcepts_corpus,
                      #goldTextConceptsAndRelations_corpus,
                      #goldTextFullNoStem_corpus,
                      goldSentFull_corpus,
                     ]

fullCorpusIndexes = [#data_orig['infracaoId'],
                     #data_orig['infracaoId'],
                     #data_orig['infracaoId'],
                     #data_orig['infracaoId'],
                     data_orig_sent_.index,
                    ]

goldCorpusIndexes = [#goldStdViolations,
                     #goldStdViolations,
                     #goldStdViolations,
                     #goldStdViolations,
                     data_orig_sent_.sort_index(level=1).loc[(slice(None), goldStdViolations),:].index,
                    ]

word2vecCossimNames = [#'trainedCbowFullCossim',
                       #'trainedCbowConceptsCossim',
                       #'trainedCbowConceptsAndRelationsCossim',
                       #'preTrainedCbow300Cossim',
                       'trainedCbowSentCossim',
                      ]

# Compute the cross-document cosine similarity using the word2vec models

word2vecModelsCossim = {}

#manager = Manager()
#word2vecModelsCossim = manager.dict()

#with concurrent.futures.ThreadPoolExecutor() as executor:

for model,corpus,goldCorpus,fullIndex,goldIndex,name in zip(word2vecModels,word2vecCorpus,word2vecGoldCorpus,
                                                            fullCorpusIndexes,goldCorpusIndexes,word2vecCossimNames):
    
    print('Calculating ', name)
    now = datetime.now() # current date and time 
    date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
    print('Start time:', date_time)

    termsim_index = WordEmbeddingSimilarityIndex(model)
    dictionary = Dictionary(corpus) # Build the term dictionary
    tfidf = TfidfModel(dictionary=dictionary) # Build the TF-IDF mapping
    bow_corpus = [dictionary.doc2bow(document) for document in corpus]

    # Build a sparse term similarity matrix using a term similarity index.
    # dictionary – A dictionary that specifies a mapping between terms and
    # the indices of rows and columns of the resulting term similarity matrix.
    # tfidf - A model that specifies the relative importance of the terms in the dictionary.
    # The columns of the term similarity matrix will be built in a decreasing order of importance of terms.

    similarity_matrix = SparseTermSimilarityMatrix(termsim_index,
                                                dictionary=dictionary,
                                                tfidf=tfidf)

    # Compute soft cosine similarity against a corpus of documents by storing the index matrix in memory.
    # corpus – A list of documents in the BoW format.
    # similarity_matrix – A term similarity matrix.

    cossim_index = SoftCosineSimilarity(tfidf[bow_corpus], similarity_matrix)

    cossim = pd.DataFrame(columns=fullIndex)

    for document in tqdm(goldCorpus):
        query_tf = tfidf[dictionary.doc2bow(document)]
        doc_similarity_scores = cossim_index[query_tf]
        try:
            cossim = cossim.append(dict(zip(cossim.columns, doc_similarity_scores)), ignore_index=True)
        except:
            cossim = cossim.append(pd.Series(0, index=cossim.columns), ignore_index=True)
    
    if isinstance(goldIndex, pd.MultiIndex):
        cossim.set_index(goldIndex, inplace=True)
        cossim = cossim.groupby(level=1).mean()
        cossim = cossim.T.groupby(level=1).mean()
    else:
        cossim['goldStdViolations'] = goldIndex
        cossim.set_index(['goldStdViolations'], inplace=True)

    word2vecModelsCossim[name] = cossim

    word2vecModelsCossim[name].to_csv("./results/" + name + ".csv")

# Compute the cross-document cosine similarity using the doc2vec models

doc2vecModels = [trainedDoc2VecSent,
                 trainedDoc2VecFull,
                 trainedDoc2VecConcepts,
                 trainedDoc2VecConceptsAndRelations]

doc2vecCorpus = [goldSentFull_corpus,
                 goldTextFull_corpus,
                 goldTextConcepts_corpus,
                 goldTextConceptsAndRelations_corpus]

doc2vecIndexes = [data_orig_sent_.sort_index(level=1).loc[(slice(None), goldStdViolations),:].index,
                  goldStdViolations,
                  goldStdViolations,
                  goldStdViolations]

doc2vecCossimNames = ['trainedDoc2VecSentCossim',
                      'trainedDoc2VecFullCossim',
                      'trainedDoc2VecConceptsCossim',
                      'trainedDoc2VecConceptsAndRelationsCossim']

doc2vecModelsEmbeddings = {}
doc2vecModelsCossim = {}

for model, corpus, index, name in zip(doc2vecModels,doc2vecCorpus,doc2vecIndexes,doc2vecCossimNames):

    print('Calculating ', name)
    now = datetime.now() # current date and time 
    date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
    print('Start time:', date_time)
    
    docEmbeddings = np.zeros((len(corpus),model.vector_size))

    for j, doc in enumerate(corpus):
        docEmbeddings[j] = model.infer_vector(doc_words=doc, steps=20, alpha=0.025)

    doc2vecModelsEmbeddings[name] = docEmbeddings

    doc2vecModelsCossim[name] = pd.DataFrame(data=cosine_similarity(doc2vecModelsEmbeddings[name]),
                                          index=index,
                                          columns=index)

    if isinstance(index, pd.MultiIndex):
        doc2vecModelsCossim[name] = doc2vecModelsCossim[name].groupby(level=1).mean()
        doc2vecModelsCossim[name] = doc2vecModelsCossim[name].T.groupby(level=1).mean()

    doc2vecModelsCossim[name].to_csv("./results/" + name + ".csv")   

#### LDA models and Cosine Similarities ####

ldaCorpus = [textFull_corpus,
             textConcepts_corpus,
             textConceptsAndRelations_corpus,
             sentFull_corpus
            ]

ldaGoldCorpus = [goldTextFull_corpus,
                 goldTextConcepts_corpus,
                 goldTextConceptsAndRelations_corpus,
                 goldSentFull_corpus,]

fullCorpusIndexes = [data_orig['infracaoId'],
                     data_orig['infracaoId'],
                     data_orig['infracaoId'],
                     data_orig_sent_.index,
                    ]

goldCorpusIndexes = [goldStdViolations,
                     goldStdViolations,
                     goldStdViolations,
                     data_orig_sent_.sort_index(level=1).loc[(slice(None), goldStdViolations),:].index,
                    ]

ldaCossimNames = ['trainedLDAFullCossim',
                  'trainedLDAConceptsCossim',
                  'trainedLDAConceptsAndRelationsCossim',
                  'trainedLDASentCossim',
                 ]

nTopicsList = [10, 20, 40, 80]

ldaModels = {}
ldaSim = {}

for corpus,goldCorpus,fullIndex,goldIndex,name in zip(ldaCorpus,ldaGoldCorpus,fullCorpusIndexes,
                                                      goldCorpusIndexes,ldaCossimNames):
    
    common_dictionary = Dictionary(corpus)
    common_corpus = [common_dictionary.doc2bow(document) for document in corpus]
    gold_corpus = [common_dictionary.doc2bow(document) for document in goldCorpus]
    for n in nTopicsList:
        ldaModels[(name, n)] = LdaModel(common_corpus, num_topics=n)

        ix = MatrixSimilarity(ldaModels[(name, n)][common_corpus])

        scores = pd.DataFrame(columns=fullIndex)
        
        for document in tqdm(gold_corpus):
            vector = ldaModels[(name, n)][document]
            docScores = ix[vector]
            try:
                scores = scores.append(dict(zip(scores.columns, docScores)), ignore_index=True)
            except:
                scores = scores.append(pd.Series(0, index=scores.columns), ignore_index=True)

        if isinstance(goldIndex, pd.MultiIndex):
            scores.set_index(goldIndex, inplace=True)
            scores = scores.groupby(level=1).mean()
            scores = scores.T.groupby(level=1).mean()
        else:
            scores['goldStdViolations'] = goldIndex
            scores.set_index(['goldStdViolations'], inplace=True)

        ldaSim[(name, n)] = scores

        ldaSim[(name, n)].to_csv("./results/" + name + "-" + str(n) + "topics.csv")

#### Computing similarites with BM25+ ranking function #####

bm25plusCorpus = [textFull_corpus,
                 textConcepts_corpus,
                 textConceptsAndRelations_corpus,
                 sentFull_corpus,]

bm25plusGoldCorpus = [goldTextFull_corpus,
                 goldTextConcepts_corpus,
                 goldTextConceptsAndRelations_corpus,
                 goldSentFull_corpus,]

bm25plusNames = ['BM25PlusFull',
                 'BM25PlusConcepts',
                 'BM25PlusConceptsAndRelations',
                 'BM25PlusSent',]

bm25plusSim = {}

for corpus,goldCorpus,fullIndex,goldIndex,name in zip(bm25plusCorpus,bm25plusGoldCorpus,fullCorpusIndexes,
                                                      goldCorpusIndexes,bm25plusNames):
    bm25plus = BM25Plus(corpus)
    
    scores = pd.DataFrame(columns=fullIndex)
    
    for document in goldCorpus:
        docScores = bm25plus.get_scores(document)
        try:
            scores = scores.append(dict(zip(scores.columns, docScores)), ignore_index=True)
        except:
            scores = scores.append(pd.Series(0, index=scores.columns), ignore_index=True)

    if isinstance(goldIndex, pd.MultiIndex):
        scores.set_index(goldIndex, inplace=True)
        scores = scores.groupby(level=1).mean()
        scores = scores.T.groupby(level=1).mean()
    else:
        scores['goldStdViolations'] = goldIndex
        scores.set_index(['goldStdViolations'], inplace=True)

    scoresMin = scores.values.min()
    scoresMax = scores.values.max()
    scores = scores.applymap(lambda x: (x-scoresMin)/(scoresMax-scoresMin))

    bm25plusSim[name] = scores

    bm25plusSim[name].to_csv("./results/" + name + ".csv")

vioCombinations = combinations(goldStdViolations, 2)
vioIndex = pd.MultiIndex.from_tuples(list(vioCombinations), names=('A', 'B'))
vioScores = pd.DataFrame(index=vioIndex)

for file in listdir('./results'):
    print('Processing file ', file)
    fileDf = pd.read_csv("./results/" + file, index_col=0)
    scoresList = []
    for pair in vioScores.index:
        scoresList.append(fileDf.loc[pair[0],str(pair[1])])
    vioScores[file.split('.')[0]] = scoresList

livia = pd.read_excel('similarity_experts_evaluation_Livia.xlsm', sheet_name='Similaridade')

scoresList = []
for pair in vioScores.index:
    try:
        scoresList.append(int(livia[(livia['Inf A']==pair[0]) & (livia['Inf B']==pair[1])]['Sim']))
    except:
        scoresList.append(int(livia[(livia['Inf A']==pair[1]) & (livia['Inf B']==pair[0])]['Sim']))
vioScores['exp1'] = scoresList

samira = pd.read_excel('similarity_experts_evaluation_Samira.xlsm', sheet_name='Similaridade')

scoresList = []
for pair in vioScores.index:
    try:
        scoresList.append(int(samira[(samira['Inf A']==pair[0]) & (samira['Inf B']==pair[1])]['Sim']))
    except:
        scoresList.append(int(samira[(samira['Inf A']==pair[1]) & (samira['Inf B']==pair[0])]['Sim']))
vioScores['exp2'] = scoresList
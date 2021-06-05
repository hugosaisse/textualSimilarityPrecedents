from datetime import datetime
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, WordEmbeddingSimilarityIndex, Word2Vec, LdaModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.keyedvectors import KeyedVectors
from gensim.similarities import SparseTermSimilarityMatrix
from gensim.similarities import SoftCosineSimilarity
from gensim.similarities import MatrixSimilarity, Similarity
import joblib
from multiprocessing import Manager
import nltk
from nltk.corpus import stopwords
import numpy as np
from os import listdir, path
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

def word2vecCossim(model,corpus,goldCorpus,fullIndex,goldIndex,name):

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

    cossim.to_csv(f"./results/{name}.csv")

def doc2vecCossim(model, corpus, index, name):

    print('Calculating ', name)
    now = datetime.now() # current date and time 
    date_time = now.strftime("%d/%m/%Y, %H:%M:%S")
    print('Start time:', date_time)
    
    docEmbeddings = np.zeros((len(corpus),model.vector_size))

    for j, doc in enumerate(corpus):
        docEmbeddings[j] = model.infer_vector(doc_words=doc, steps=20, alpha=0.025)

    doc2vecModelsCossim = pd.DataFrame(data=cosine_similarity(docEmbeddings),
                                          index=index,
                                          columns=index)

    if isinstance(index, pd.MultiIndex):
        doc2vecModelsCossim = doc2vecModelsCossim.groupby(level=1).mean()
        doc2vecModelsCossim = doc2vecModelsCossim.T.groupby(level=1).mean()

    doc2vecModelsCossim.to_csv("./results/" + name + ".csv")

#engine = sal.create_engine('mssql+pyodbc://LAPTOP-NNDGMEMB/beth?driver=ODBC+Driver+13+for+SQL+Server?Trusted_Connection=yes')
#conn = engine.connect()
#data = pd.read_sql_table('violations_content', conn)
data = pd.read_csv('../data.csv')

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

data_orig_sent_ = data_orig_sent.sort_values(by=['infracaoId']).reset_index(drop=True)
data_orig_sent_ = data_orig_sent_.set_index([data_orig_sent_.index, 'infracaoId'])

#### TF-IDF vectorization and Cosine Similarities ####

#vectorizer

nGramRangeDict = {'UniBi': (1,2),
                  'BiTri': (2,3)
                 }

docRepDict = {'Full': data_orig['stemTextFull'],
              'Con': data_orig['stemTextConcepts'],
              'ConRel': data_orig['stemTextConceptsAndRelations'],
              'Sent': data_orig_sent_['stemSentFull']
             }

for key, value in nGramRangeDict.items():

    print(f'Creating TF-IDF vectorizer for ngramrange={str(value)}...')
    
    vect = TfidfVectorizer(lowercase=False, max_df=0.8, ngram_range=value)
    vect = vect.fit(docRepDict['Full'])
    
    for k, v in docRepDict.items():

        print(f'Transforming {k} document representation...')

        tfidf = vect.transform(v)
        #tfidf = vect.fit_transform(v)
        
        print(f'Calculating cosine similarity for {k} document representation...')

        if k == 'Sent':
            cossim = cosineSimilarityByViolation(tfidf)
        else:
            cossim = pd.DataFrame(data=cosine_similarity(tfidf),
                                index=data_orig['infracaoId'].tolist(),
                                columns=data_orig['infracaoId'].tolist())
        
        cossim.to_csv(f'./results/tfidf_{key}_{k}.csv')

#### Creating the corpora for training W2V, D2V, LDA and BM25 models ####

docRepDict2 = {'FullNoStem': [document.split() for document in data_orig['regexCleanedText']],
               'Full': [document.split() for document in data_orig['stemTextFull']],
               'Con': [document.split() for document in data_orig['stemTextConcepts']],
               'ConRel': [document.split() for document in data_orig['stemTextConceptsAndRelations']],
               'Sent': [document.split() for document in data_orig_sent_['stemSentFull']]
              }

textFullNoStem_corpus = [document.split() for document in data_orig['regexCleanedText']]

textFull_corpus = [document.split() for document in data_orig['stemTextFull']]
textConcepts_corpus = [document.split() for document in data_orig['stemTextConcepts']]
textConceptsAndRelations_corpus = [document.split() for document in data_orig['stemTextConceptsAndRelations']]
sentFull_corpus = [document.split() for document in data_orig_sent_['stemSentFull']]

#### Creating corpora that only includes the violations included in the gold standard ####

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

docRepDict3 = {'FullNoStem': goldTextFullNoStem_corpus,
               'Full': goldTextFull_corpus,
               'Con': goldTextConcepts_corpus,
               'ConRel': goldTextConceptsAndRelations_corpus,
               'Sent': goldSentFull_corpus
              }

#### Word2Vec & Doc2Vec Models and Cosine Similarities ####

# Training Word2Vec model from the existing complete corpus

vectorSize = {
    'v100': 100,
    'v300': 300,
}

method = {
    'cbow': 0,
    'skipGram': 1,
}

for k1, v1 in docRepDict2.items():

    for k2, v2 in vectorSize.items():

        for k3, v3 in method.items():

            print(f'Training Word2Vec for corpus {k1}, dimension {str(v2)} and method {k3} ...')

            if k1 != 'FullNoStem':
                word2vec = Word2Vec(sentences=v1, size=v2, workers=8, min_count=3, sg=v3)
                word2vec.save(f'./models/trained_{k3}_{k2}_{k1}.model')

            if k1 == 'FullNoStem':
                word2vec = KeyedVectors.load_word2vec_format(f'./models/{k3}_s{str(v2)}.txt', binary=False, unicode_errors='ignore')
                word2vec.save(f'./models/preTrained_{k3}_{k2}_{k1}.model')

# Loading Word2Vec models from the models folder

# Vectorizing from the pretrained NILC-USP models

for file in [f for f in listdir('./models/pretrained') if f.endswith('.model')]:

    absFile = path.abspath('./models/pretrained/'+file)
    
    name = str(file).split('.')[0] + '_Cossim'
    fullIndex = data_orig['infracaoId']
    goldIndex = goldStdViolations
    modelWv = KeyedVectors.load(absFile)
    corpus = docRepDict2['FullNoStem']
    goldCorpus = docRepDict3['FullNoStem']

    print(f"Calculating cossim for model {str(file).split('.')[0]} ...")
   
    word2vecCossim(modelWv,corpus,goldCorpus,fullIndex,goldIndex,name)

# Vectorizing from the corpora trained models

for file in [f for f in listdir('./models/') if f.endswith('.model')]:

    absFile = path.abspath('./models/'+file)
    
    name = str(file).split('.')[0] + '_Cossim'
    fullIndex = data_orig['infracaoId']
    goldIndex = goldStdViolations

    if 'Full' in str(file):
        model = Word2Vec.load(absFile)
        modelWv = model.wv
        corpus = docRepDict2['Full']
        goldCorpus = docRepDict3['Full']

    elif 'ConRel' in str(file):
        model = Word2Vec.load(absFile)
        modelWv = model.wv
        corpus = docRepDict2['ConRel']
        goldCorpus = docRepDict3['ConRel']
    
    elif 'Con' in str(file):
        model = Word2Vec.load(absFile)
        modelWv = model.wv
        corpus = docRepDict2['Con']
        goldCorpus = docRepDict3['Con']
    
    elif 'Sent' in str(file):
        model = Word2Vec.load(absFile)
        modelWv = model.wv
        corpus = docRepDict2['Sent']
        goldCorpus = docRepDict3['Sent']
        fullIndex = data_orig_sent_.index
        goldIndex = data_orig_sent_.sort_index(level=1).loc[(slice(None), goldStdViolations),:].index

    print(f"Calculating cossim for model {str(file).split('.')[0]} ...")
   
    word2vecCossim(modelWv,corpus,goldCorpus,fullIndex,goldIndex,name)

# Vectorizing from the full corpus trained models

for file in [f for f in listdir('./models/') if f.endswith('Full.model')]:
  
    absFile = path.abspath('./models/'+file)
    
    fullIndex = data_orig['infracaoId']
    goldIndex = goldStdViolations
    model = Word2Vec.load(absFile)
    modelWv = model.wv

    print(f"Calculating cossim for model {str(file).split('.')[0]} ...")

    for keyCorpus, keyGoldCorpus in zip(list(docRepDict2.keys())[1:], list(docRepDict3.keys())[1:]):
        
        name = str(file).split('.')[0] + '_' + str(keyCorpus) + '_Cossim'

        if path.exists("./results/" + name + ".csv"):
            continue

        corpus = docRepDict2[keyCorpus]
        goldCorpus = docRepDict3[keyCorpus]

        if keyCorpus == 'Sent':
            
            fullIndex = data_orig_sent_.index
            goldIndex = data_orig_sent_.sort_index(level=1).loc[(slice(None), goldStdViolations),:].index

        

        print(f"Calculating cossim for corpus {str(keyCorpus)} ...")
    
        word2vecCossim(modelWv,corpus,goldCorpus,fullIndex,goldIndex,name)


# Training Doc2Vec models 

docRepDict4 = {'Full': [document.split() for document in data_orig['stemTextFull']],
               'Con': [document.split() for document in data_orig['stemTextConcepts']],
               'ConRel': [document.split() for document in data_orig['stemTextConceptsAndRelations']],
               'Sent': [document.split() for document in data_orig_sent_['stemSentFull']]
              }

vectorSize = {
    'v100': 100,
    'v200': 200,
    'v300': 300,
}

for k1, v1 in docRepDict4.items():

    for k2, v2 in vectorSize.items():

        print(f'Training Doc2Vec for corpus {k1}, dimension {str(v2)} ...')

        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(v1)]
        doc2vec = Doc2Vec(documents, min_count=3, workers=8, size=v2)
        doc2vec.save(f'./models/doc2vec/doc2vec_{k2}_{k1}.model')

# Loading Doc2Vec models from the models/doc2vec folder
# and compute the cross-document cosine similarity using the doc2vec models

for file in [f for f in listdir('./models/doc2vec') if f.endswith('.model')]:

    absFile = path.abspath('./models/doc2vec/'+file)
    
    name = str(file).split('.')[0] + '_Cossim'
    index = goldStdViolations
    model = Doc2Vec.load(absFile)

    if 'Full' in str(file):
        corpus = goldTextFull_corpus

    elif 'ConRel' in str(file):
        corpus = goldTextConceptsAndRelations_corpus
    
    elif 'Con' in str(file):
        corpus = goldTextConcepts_corpus
    
    elif 'Sent' in str(file):
        corpus = goldSentFull_corpus
        index = data_orig_sent_.sort_index(level=1).loc[(slice(None), goldStdViolations),:].index

    print(f"Calculating cossim for model {str(file).split('.')[0]} ...")
   
    doc2vecCossim(model, corpus, index, name)

#### LDA models and Cosine Similarities ####

fullCorpus = [textFull_corpus,
              textConcepts_corpus,
              textConceptsAndRelations_corpus,
              sentFull_corpus]

fullCorpusIndexes = [data_orig['infracaoId'],
                     data_orig['infracaoId'],
                     data_orig['infracaoId'],
                     data_orig_sent_.index,
                    ]

ldaGoldCorpus = [goldTextFull_corpus,
                 goldTextConcepts_corpus,
                 goldTextConceptsAndRelations_corpus,
                 goldSentFull_corpus,]

goldCorpusIndexes = [goldStdViolations,
                     goldStdViolations,
                     goldStdViolations,
                     data_orig_sent_.sort_index(level=1).loc[(slice(None), goldStdViolations),:].index,
                    ]

ldaCossimNames = ['LDAFullCossim',
                  'LDAConceptsCossim',
                  'LDAConceptsAndRelationsCossim',
                  'LDASentCossim',
                 ]

nTopicsList = [10, 20, 40, 80]

ldaModels = {}
ldaSim = {}

common_dictionary = Dictionary(textFull_corpus)
common_corpus = [common_dictionary.doc2bow(document) for document in textFull_corpus]

for corpus,index,goldCorpus,goldIndex,name in zip(fullCorpus,fullCorpusIndexes,ldaGoldCorpus,goldCorpusIndexes,ldaCossimNames):
    
    #common_dictionary = Dictionary(corpus)
    #common_corpus = [common_dictionary.doc2bow(document) for document in corpus]
    
    gold_corpus = [common_dictionary.doc2bow(document) for document in goldCorpus]
    for n in nTopicsList:
        ldaModels[(name, n)] = LdaModel(common_corpus, num_topics=n)

        ix = MatrixSimilarity(ldaModels[(name, n)][common_corpus])

        scores = pd.DataFrame(columns=index)
        
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

bm25plus = BM25Plus(textFull_corpus)

for corpus,goldCorpus,fullIndex,goldIndex,name in zip(bm25plusCorpus,bm25plusGoldCorpus,fullCorpusIndexes,
                                                      goldCorpusIndexes,bm25plusNames):
    #bm25plus = BM25Plus(corpus)
    
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

    #scoresMin = scores.values.min()
    #scoresMax = scores.values.max()
    #scores = scores.applymap(lambda x: (x-scoresMin)/(scoresMax-scoresMin))

    bm25plusSim[name] = scores

    bm25plusSim[name].to_csv("./results/" + name + ".csv")

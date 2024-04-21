from itertools import combinations
import numpy as np
from os import listdir, walk
import pandas as pd
from tqdm import tqdm

def topNSimilarViolations(simArray, n, violationId, threshold = 0):
    """
    Returns the 'n' most similar violations for a 'violationId'
    'simArray' is a similarity array obtained from a model
    """
    filteredSimArray = simArray[vioScores.index.isin([violationId], level=0) | vioScores.index.isin([violationId], level=1)]
    topSim = filteredSimArray[filteredSimArray > threshold].nlargest(n)
    l = []
    for tup in topSim.index:
        for el in tup:
            if el != violationId:
                l.append(el)
    topSim.index = l

    return topSim

#### Retrieve the violations used in the experts evaluation (gold-standard)

goldStdViolations = pd.read_excel('vioWithData.xlsx')['infracaoId'].sort_values(ascending=True).to_list()

#### Generate the possible combinations with the violations
#### and create the vioScores dataframe to store the similarity scores
#### for the pairs and different models

vioCombinations = combinations(goldStdViolations, 2)
vioIndex = pd.MultiIndex.from_tuples(list(vioCombinations), names=('A', 'B'))
vioScores = pd.DataFrame(index=vioIndex)

#### For each file in the results file one column will be generated with the similarity scores

for file in listdir('./results'):
    print('Processing file ', file)
    fileDf = pd.read_csv("./results/" + file, index_col=0)
    scoresList = []
    for pair in vioScores.index:
        scoresList.append(fileDf.loc[pair[0],str(pair[1])])
    vioScores[file.split('.')[0]] = scoresList

#### For each file in the results folder

modelDescr = pd.DataFrame(columns=['Model',
                                   'Training Corpus',
                                   'Evaluation Corpus',
                                   'Topics',
                                   'Dimensions',
                                   'Algorithm',
                                   'N-Grams'])

for t in walk('./results'):
    folder = t[0]
    subfolders = t[1]
    files = t[2]

    for file in files:
        print(f'Processing file {folder}/{file}')
        fileDf = pd.read_csv(f'{folder}/{file}', index_col=0)
        scoresList = []
        for pair in vioScores.index:
            scoresList.append(fileDf.loc[pair[0],str(pair[1])])
        vioScores[folder+file.split('.')[0]] = scoresList
        
        descrList = []
        #Model
        if any(x in file for x in ['skipGram','cbow','preTrained']):
            descrList.append('Word2Vec')
        elif 'doc2vec' in file:
            descrList.append('Doc2Vec')
        elif 'LDA' in file:
            descrList.append('LDA')
        elif 'tfidf' in file:
            descrList.append('TF-IDF')
        elif 'BM25' in file:
            descrList.append('BM25')
        else:
            descrList.append('-')
        
        #Training corpus
        if 'preTrained' in file:
            descrList.append('Full (No Stemming)')
        elif 'trainedOnFullCorpus' in folder:
            descrList.append('Full')
        elif 'trainedOnCorpus' in folder:
            if any(x in file for x in ['ConRel','ConceptsAndRelations']):
                descrList.append('Concepts & Relations')
            elif any(x in file for x in ['Con','Concepts']):
                descrList.append('Concepts')
            elif 'Sent' in file:
                descrList.append('Sentences')
            elif 'Full' in file:
                descrList.append('Full')
            else:
                descrList.append('-')
        else:
            descrList.append('-')
        
        #Evaluation corpus
        if 'preTrained' in file:
            descrList.append('Full (No Stemming)')
        elif any(x in file for x in ['ConRel','ConceptsAndRelations']):
            descrList.append('Concepts & Relations')
        elif any(x in file for x in ['Con','Concepts']):
            descrList.append('Concepts')
        elif 'Sent' in file:
            descrList.append('Sentences')
        elif 'Full' in file:
            descrList.append('Full')
        else:
            descrList.append('-')
        
        #Topics
        pos = file.find('topics')
        if pos != -1:
            descrList.append(file[pos-2:pos])
        else:
            descrList.append('-')
        
        #Dimensions
        pos = file.find('00_')
        if pos != -1:
            descrList.append(file[pos-1:pos+2])
        else:
            descrList.append('-')
        
        #Algorithm
        if 'skipGram' in file:
            descrList.append('Skip-Gram')
        elif 'cbow' in file:
            descrList.append('CBOW')
        else:
            descrList.append('-')

        #N-Grams
        if 'UniBi' in file:
            descrList.append('Unigrams and Bigrams')
        elif 'BiTri' in file:
            descrList.append('Bigrams and Trigrams')
        else:
            descrList.append('-')

        modelDescr.loc[len(modelDescr)] = descrList

#### include the experts evaluations for each pair in the vioScores dataframe

evaluations = []
for file in listdir('./goldStandardViolations_/expertsEvaluations'):
    evaluations.append(pd.read_excel("./goldStandardViolations_/expertsEvaluations/" + file, sheet_name='Similaridade'))

for i, evaluation in enumerate(evaluations):
    scoresList = []
    for pair in vioScores.index:
        try:
            scoresList.append(int(evaluation[(evaluation['Inf A']==pair[0]) & (evaluation['Inf B']==pair[1])]['Sim']))
        except:
            scoresList.append(int(evaluation[(evaluation['Inf A']==pair[1]) & (evaluation['Inf B']==pair[0])]['Sim']))
    vioScores['exp'+str(i)] = scoresList

#### generate one column for the mean between the experts scores

vioScores['expMean'] = np.mean(vioScores[[col for col in vioScores.columns if 'exp' in col]], axis=1)

#### generate dataframe for the first classification experiment
#### targetOne:
#### any expert score mean different from zero will be treated as having some similarity (1)
#### if the expert score mean is equal to zero the pair will be treated as having no similarity (0)
#### targetTwo:
#### any expert score mean of 2.5 or greater will be treated as having similarity (1)
#### otherwise it will be treated as having no similarity (0)

def anyIsDiffFromZero(row):
    if (row['exp1']>=3 or row['exp2']>=3 or row['exp0']>=3) != 0:
        return 1
    else:
        return 0

experiments = pd.DataFrame(index=vioIndex)
experiments['targetOne'] = vioScores['expMean'].apply(lambda x: 1 if x>0 else 0)
experiments['targetTwo'] = vioScores['expMean'].apply(lambda x: 1 if x>2.5 else 0)
experiments['targetThree'] = vioScores['expMean'].apply(lambda x: 1 if x>4 else 0)


#### modelOne:
#### If the violation B appears in the top 5 similar violations for violation A
#### the modelOne value will be set to 1, otherwise it will be set to zero

def isInTopN(row, modelName, N):
    """
    Checks if a violation B is in the top N similar violations to violation A.
    To be applied rowwise on a dataframe.
    Returns 1 if yes. Otherwise returns 0.
    """
    a = topNSimilarViolations(vioScores[modelName], N, row.name[0])
    if row.name[1] in a.index:
        return 1
    else:
        return 0

def calculateMetrics(vioScores, expColumn):

    results = pd.DataFrame(index=vioScores.columns[:-4], columns=['mAR', 'recall@5', 'mAP', 'precision@5'])
    for modelName in vioScores.columns[:-4]:
        avgRecallList = []
        avgPrecisionList = []
        vioWithNoSimilar = set()
        for N in tqdm(range(1,51)):
            recallList = []
            precisionList = []
            for violation in goldStdViolations:
                topNSimilar_model = topNSimilarViolations(vioScores[modelName], N, violation)
                topNSimilar_experts = topNSimilarViolations(vioScores[expColumn], N, violation)
                totalSimilar_experts = topNSimilarViolations(vioScores[expColumn], 50, violation)
                Num = len(set(topNSimilar_model.index).intersection(set(totalSimilar_experts.index)))
                recallDen = len(totalSimilar_experts.index)
                precisionDen = len(topNSimilar_model.index)
                if recallDen == 0:
                    recall = 1
                    vioWithNoSimilar.add(violation)
                else:
                    recall = Num/recallDen
                if precisionDen == 0 & len(topNSimilar_experts.index) != 0:
                    precision = 0
                elif precisionDen == 0 & len(topNSimilar_experts.index) == 0:
                    precision = 1
                else:
                    precision = Num/precisionDen
                recallList.append(recall)
                precisionList.append(precision)

            avgRecall = np.mean(recallList)
            avgPrecision = np.mean(precisionList)
            avgRecallList.append(avgRecall)
            avgPrecisionList.append(avgPrecision)
            if N == 5:
                results.at[modelName, 'recall@5'] = avgRecall
                results.at[modelName, 'precision@5'] = avgPrecision
        meanAvgRecall = np.mean(avgRecallList)
        meanAvgPrecision = np.mean(avgPrecisionList)
        results.at[modelName, 'mAR'] = meanAvgRecall
        results.at[modelName, 'mAP'] = meanAvgPrecision
    results['f1'] = (2*results['mAR']*results['mAP'])/(results['mAR']+results['mAP'])

    return results

expColumns = vioScores.columns[-4:].tolist()
expResults = {}

for expColumn in expColumns:
    result = calculateMetrics(vioScores, expColumn)
    result = pd.concat([modelDescr.set_index(result.index), result], axis=1)
    expResults[expColumn] = result
    result.to_csv(expColumn + '.csv')

# Remove duplicate models (Trained on full corpus and evaluated on full corpus)
# Find duplicate rows
for file in [x for x in listdir('./') if ('exp' in x) and ('.csv' in x)]:
    expDf = pd.read_csv(file, index_col=0)
    expDf.drop_duplicates(subset=['Model', 'Training Corpus', 'Evaluation Corpus',
                    'Topics', 'Dimensions', 'Algorithm', 'N-Grams'], inplace=True)
    expDf.to_csv(file)
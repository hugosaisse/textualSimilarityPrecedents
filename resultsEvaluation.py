from itertools import combinations
import numpy as np
from os import listdir
import pandas as pd

goldStdViolations = pd.read_excel('vioWithData.xlsx')['infracaoId'].sort_values(ascending=True).to_list()

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

evaluations = []
for file in listdir('./goldStandardViolations/expertsEvaluations'):
    evaluations.append(pd.read_excel("./goldStandardViolations/expertsEvaluations/" + file, sheet_name='Similaridade'))

for i, evaluation in enumerate(evaluations):
    scoresList = []
    for pair in vioScores.index:
        try:
            scoresList.append(int(evaluation[(evaluation['Inf A']==pair[0]) & (evaluation['Inf B']==pair[1])]['Sim']))
        except:
            scoresList.append(int(evaluation[(evaluation['Inf A']==pair[1]) & (evaluation['Inf B']==pair[0])]['Sim']))
    vioScores['exp'+str(i)] = scoresList

vioScores['expMean'] = np.mean(vioScores[[col for col in vioScores.columns if 'exp' in col]], axis=1)


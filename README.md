# Project Title

This project is related to the paper "Textual similarity for legal precedents discovery: assessing the performance of machine learning techniques in an administrative court" which is currently under review at the International Journal of Information Management Data Insights.

## Project Structure

- `goldStandardViolations`: This directory contains the code to create a random set of case pairs.
- `goldStandardViolations/expertsEvaluations`: This directory contains expert scores to the case pairs.
- `models`: These are the models used in the paper, both pre-trained and fine-tuned models.
- `results`: These files contain results per model assembly.
- `documentSimilarity.py`: The code used to compute the similarity between the case pairs employing the models and assemblies under analysis.
- `resultsEvaluation.py`: The code used to compute the performance metrics considering the results.

Please note that some huge files and directories are ignored by Git as specified in the `.gitignore` file. At the same time, the content of infractions was not included in the repository due to the nature of the data, which includes confidential information about people and companies and the penalties issued by SUSEP. It could violate personal, commercial, and competitive confidentiality.

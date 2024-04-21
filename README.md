# Project Title

This project is related to the paper "Textual similarity for legal precedents discovery: assessing the performance of machine learning techniques in an administrative court" which is currently under review at the International Journal of Information Management Data Insights.

## Project Structure

- `goldStandardViolations`: This directory contains the code used to create a random set of case pairs.
- `goldStandardViolations/expertsEvaluations`: This directory contains the scores given by the experts to the case pairs.
- `models`: These are the models used in the paper, both pre-trained and fine-tuned models.
- `results`: These are the files containing results per model assembly.
- `documentSimilarity.py`: The code used to compute similarity between the case pairs employing the models and assemblies under analysis.
- `resultsEvaluation.py`: The code used to compute the performance metrics considering the results.

Please note that the some huge files and directories are ignored by Git as specified in the `.gitignore` file.

## Getting Started

To get started with this project, clone the repository and install the required dependencies.

## Contributing

We welcome contributions! Please see our contributing guidelines for details.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

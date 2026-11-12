# kaggle_ufg

My scripts for a Kaggle competition run during the master's program at the Federal University of
Goiás (UFG) in 2019: predict the category of a marketplace product from its title, in Portuguese.

- 33,500 training titles in 83 categories (`data/train.csv`)
- 16,500 test titles (`data/test.csv`)

Example: `Copo Térmico Plástico Os Jetsons Família Espacial` → `THERMAL_CUPS_AND_TUMBLERS`

## Approach

- Text cleaning and stopword removal (`util.py`, `stopwords.py`)
- TF-IDF and count vectors (`classificacao.py`, `pipeline_simples.py`)
- Random Forest, SGD, gradient boosting, KNN and voting ensembles
- Class imbalance handled with Tomek links, SMOTE and undersampling
- Side studies: LDA topics (`LDA.py`, `topic.py`), k-means (`kmeans.py`), outliers (`outlier.py`)
- `submissao.py` trains the final model and writes the submission; past submissions are in `sumissoes/`

This is exploratory competition code, kept as it was written.

## Requirements

```bash
pip install pandas numpy scikit-learn imbalanced-learn nltk
```

# Canadian-caribou-zoo
## Sub-title: python scripts to predict ecotypes of caribou samples coming from Canadian zoo.
These scripts support the publication from Luzuriaga-Aveiga et coll: Integrating genomic tools into ex situ conservation management of caribou (Rangifer tarandus) in
Canada.
These scripts are provided "as is," with no warranty or guarantee of any kind for their use, in whole or in part.

## Summary:
The model to predict ecotypes from genomic data was developped and optimized using the script _ML-ecotype3.py_. The development was based on a typical machine-learning approach where the data set was first split into test and learning sets. Then a cross-validation procedure was used to train the model and test various hyper-parameters. The final model was saved and used in the second analysis (scripted in _predict-zoo-samples_) to predict the ecotype of zoo samples based on genotypes.

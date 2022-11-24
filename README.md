# GA Pipeline to Design Materials for Tandem Organic Solar Cells

This repository contains the data files and analysis notebooks to support the paper "Designing Efficient Tandem Organic Solar Cells with Machine Learning and Genetic Algorithms."

The PCE prediction model, known as OPEP2, was trained on a dataset containing 1,001 unique donor/acceptor pairs. This dataset can be found here: `OPEP2/filtered_dataset.csv`
The unfiltered dataset containing 1,225 donor/accpetor pairs can be found here: `OPEP2/Experimental_data.csv` 


The OPEP2 ML model to predict PCE can be found here: `OPEP2/Prediction_models/ensemble_rf_ann_mfcounts_standardized_above10_fixed.pkl`

To retrain the model with new data, a detailed Jupyter notebook can be found here: `OPEP2/train_OPEP2.ipynb`


### Versions of libraries
* python 3.9
* scikit-learn 1.0.2
* rdkit 2021.09.4
* pandas 1.4.0
* scipy 1.7.3
* xgboost 1.5.0
* statsmodels 0.13.2
* lightgbm 3.2.1
* numpy 1.20.0
* matplotlib 3.5.1

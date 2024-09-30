# CML_processing_by_ML
Codes to train a Fully Conv. Net. (FCN) on 15 sec 2-channels attenuations.

- notebooks:
    - illustrates how 1D causal FCN work on simulated data (training_on_simulated_data.ipynb)
    - displays fcn technical characteristics, comprising causality (model_characteristics.ipynb)
    - shows examples of training curves on the VAL INTRA set (training_curves.ipynb)
    - see how figures of the poster have been obtained (training_on_simulated_data.ipynb)

- src:
    - utils: see transforms, pytorch Datasets objects, architecture details and cost functions 
    - train_1GPU: the simplest training procedure

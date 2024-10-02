# CML_processing_by_ML
Github repository referenced by the poster "Raincell project: Commercial Microwave Links to estimate rainfall at Météo-France" (ERAD 2024)

- notebooks:
    - **training_on_simulated_data**: illustrates how 1D causal FCN work on simulated data (training_on_simulated_data.ipynb)
    - **model_characteristics**: displays fcn technical characteristics, comprising causality (model_characteristics.ipynb)
    - **training_curves**: shows examples of training curves on the VAL INTRA set (training_curves.ipynb)
    - **evaluation_poster_val_intra**: see how figures of the poster have been obtained (training_on_simulated_data.ipynb)

- src:
    - **utils**: see transforms, pytorch Datasets objects, architecture details and cost functions 
    - **train_1GPU**: the simplest training procedure

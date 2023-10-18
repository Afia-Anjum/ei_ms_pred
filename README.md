Thesis title: Application of Machine Learning Towards Compound Identification through Gas Chromatography Retention Index (RI) and Electron Ionization Mass Spectrometry (EI-MS) Predictions
This is the github repo for chapter 3 "Prediction of Electron-Ionization Mass Spectra (EI-MS) by Leveraging Composite Graph Neural Networks (GNN) and a Support Vector Regressor (SVR)"


These are the require packages and set up for a conda environment (can be slightly different depending on system).

```
conda create -c rdkit -n prop_predictor rdkit
source activate prop_predictor
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install scikit-learn tqdm
conda activate prop_predictor

#to deactivate an environment:
conda deactivate

```

Add the repo to PYTHONPATH:


To compute the shortest paths, run the following:
```
python preprocess/shortest_paths.py -data_dir path_to_data
```

To compute the splitting 
```
python parse/split_data.py -data_path data/rough_eims/raw.csv -output_dir data/rough_eims -multi
```
To run the models:
```
dataset=path_to_dataset
python train/train_prop.py -cuda \
    -data $dataset -loss_type mse \
    -max_grad_norm 10 -batch_size 50 -num_epochs 100 \
	-output_dir output_test/sol_transformer -n_rounds 10 \
	-model_type transformer -hidden_size 160 \
	-p_embed -ring_embed -max_path_length 3 -lr 5e-4 \
	-no_share -n_heads 2 -d_k 80 -dropout 0.2
```

To evaluate the model using new test set
CUDA_VISIBLE_DEVICES=$device python train/train_test.py -cuda -data $dataset -loss_type mae -max_grad_norm 10 -batch_size 100 -batch_splits 2 -num_epochs 1000 -output_dir output_test/understdnp_transformer -n_rounds 1 -model_type transformer -hidden_size 160 -p_embed -ring_embed -max_path_length 3 -lr 5e-4 -no_share -n_heads 2 -d_k 80 -dropout 0.2


############################################################MODEL: EI-MS alpha########################################################
-> Model is placed in the /models/EI-MS_alpha folder


############################################################MODEL: EI-MS beta#########################################################
-> Model is placed in the /models/EI-MS_beta folder
-> The models created from the high m/z spectra set and the low m/z spectra set are convoluted together.

############################################################MODEL: EI-MS gamma#########################################################
-> Model is the convolution of the low m/z (<=130 Da) EI-MS beta predictor with the high m/z portion (>130 Da) of CFM-EI(Felicity et. al, 2016) that uses a probabilistic graphical model.

############################################################MODEL: MIIP###############################################################
-> Model is placed in the path /models/MIIP_SVR.py 
-> The finalized model will be available upon request

############################################################MODEL: EI-MS delta########################################################
-> Model is the convolution of the EI-MS alpha and MIIP
-> The convolution is scripted in the python file /models/post_processed_models.py

############################################################MODEL: EI-MS epsilon######################################################
-> Model is the convolution of the EI-MS beta and MIIP
-> The convolution is scripted in the python file /models/post_processed_models.py

############################################################MODEL: NEIMS##############################################################
-> The NEIMS program was run for all the compounds in the held out test sets. Various pre-processing were needed and the script used to make the processing are placed in the path /utils/general_preproccing_files.py

############################################################MODEL: RASSP##############################################################
-> The RASSP program was run for all the compounds in the held out test sets. Various pre-processing were needed and the script used to make the processing are placed in the path /utils/general_preproccing_files.py

############################################################MODEL: PeakAnnotator (subformula generation)##############################
-> The program is placed under the path /models/peak_annotator.py

############################################################MODEL: Adjusted NEIMS with MIIP and PA####################################
-> This model is the combination of the NEIMS, MIIP and PA program together. 
-> The convolution is also scripted in the python file /models/post_processed_models.py

#To build a set of training set tuple with m/z, intensity and subformula:
-> annotateNISTgroundEIwithformula.py will be used. This incorporates the annotation tehcniques followed during the peak annotation procedure of PeakAnnotator.

ALL Data (used in all these experiments are from NIST20 and NIST23 and may be available upon request).
ALL Model (generated through these experiments are currently in private mode and may be available upon request).
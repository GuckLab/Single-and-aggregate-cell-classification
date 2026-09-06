# Single-cell-and-aggregate-classification
This repository contains Python code for the classification of single blood cells and blood cell aggregates in images acquired with a deformability cytometry device.

The repository accompanies the following paper: Zingman et al. Multi-label versus multi-class classification of blood cells and their aggregates in
microfluidic channels, 2026, which is available at Arxiv:
**Project organization**

**Requirements** 

**Installation**


**Setting up the dataset**

The accompanying data can be downloaded from [Open Science Framework](https://osf.io/3zkvw/). The data can be located in the project folder *data* 


**Training and evaluation**

To train the model, run the following command in the terminal:

```
python train.py --param_file  ./configurations/configuration_file.yaml --data_path ./data/  --data_path_gmm_eval_in ./data/WBCtest_data/ --data_path_gmm_eval_out ./predictions_temp/
```

The evaluation of the classification performance on the *test_data* and *WBCtest_data* is automatically done after training has been finished.
The evaluation of already trained model can also be performed separately by running the following command in the terminal:

```
python evaluate.py --path_in ./data/ --model model_Run_ID --path_in_gmm ./data/WBCtest_data/ --path_out_gmm_pred ./predictions_temp/
```
model_Run_ID is available from mlflow dashboard after training the model. It is also saved in run_uuid.txt file in the project folder, after model was trained.



The results will be tracked in the folder *./mlflowruns/*, which is defined in .env file. You can change the path to the mlflow runs folder by changing the value of the variable *MLFLOW_TRACKING_URI* in the .env file.



**Results**





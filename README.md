# cell-cycle-classification

## Environment 
 * `requirements.txt`

## Directory
* Modify saving directory like checkpoint, record and model via `config.py`

## Step 1: Mitochondrai and nucleus prediction from brightfield
 ### Data
  * `/Data/Hoechst`
 ### Training
 * `cd train`
 * `python train_predict_mito_nuclei.py`

 ### Result    
 * UNet Training result     
    - **Mitochondria**  
        - Training Pearson Correlation Coefficient: 0.78
        - Validation Pearson Correlation Coefficient: 0.72  
    - **Mitochondria**  
        - Training Pearson Correlation Coefficient: 0.85
        - Validation Pearson Correlation Coefficient: 0.83    

## Step 2: Cell cycle classification from brightfield/predicted mitochondria/predicted nucleus (This part is still continuing!)
 ### Data
  * `/Data/FUCCI`
 ### Training
 * `cd train`
 * `python train_cellcycle_phase.py`


### Plot
 * Plot training process through `plot/plot_result.py`
 * Predict through `plot/predict.py`

### Result
 * Few results saved in `Mito&Nuclei_prediction` & `GFP&RFP_prediction`

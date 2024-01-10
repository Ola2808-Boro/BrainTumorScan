
# Brain tumor segmentation/classification from MRI images

## Project goal
The aim of the project is to perform segmentation based on . The U-net model was used for this process, which is located in the segmentation.py file along with the entire segmentation process. Classifications were also performed on the combined sets available on Kaggle() and on the set of augmented data. Experiments were carried out on three moodels: EfficientNetB3, Resnet50 and on a model with its own network architecture. The classification is located in the classification.py file.


## Project to do list

- [x]  review of the literature
- [x]  defining the experiments being performed
- [x]  connection of 3 databases
- [x]  generating a new database (photos + masks)
- [x]  training a model for U-net segmentation
- [x]  training 3 models for classification
- [x]  creating an application for segmentation and prediction of brain cancer from MRI images

## Tech Stack

Python, PyTorch, TensorFlow, Flask 


## Demo

[![Watch the video](https://github.com/Ola2808-Boro/BrainTumorScan/blob/main/BRAINTumorScan.png)](https://youtu.be/sUmUPkeFvuw)
## Results

The results in the tables are given for the most recent epochs.

### EfficientNetB3

|Epochs| Train loss  | Train accuracy | Validation loss | Validation accuracy|
| ------------- | ------------- | ------------- | ------------- | ------------- |
| 5  |0.7428  | 0.8491  | 0.7388  | 0.8291  |
| 54  |0.0993 | 0.9869  |0.1105  |0.9802  |



### Resnet50

|Epochs| Train loss  | Train accuracy | Validation loss | Validation accuracy|
| ------------- | ------------- | ------------- | ------------- | ------------- |
| 5  |0.6404  | 0.7941  | 1.3883  | 0.6056  |
| 61  |0.2053  | 0.9510  |0.2515  |0.9329  |



### Own CNN architecture

|Epochs| Train loss  | Train accuracy | Validation loss | Validation accuracy|
| ------------- | ------------- | ------------- | ------------- | ------------- |
| 5  |0.2620  | 0.9278  | 0.3960  | 0.8757  |
| 34  |0.0116  | 0.9987  |0.0378  |0.9931  |



## Authors

- [@Aleksandra Borowska](https://github.com/Ola2808-Boro)
- [@Patryk Spierewka](https://github.com/PatrykSpierewka)


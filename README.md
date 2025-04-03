DEEP LEARNING PROJECT
GREY BALL INSERTION

Dataset:
- https://www2.cs.sfu.ca/~colour/data2/DRONE-Dataset/
- 
Pyton files:
- DataImport: Downloads all the necessary images needed for the project into 2 new folders: scenes and scenes_shots.
- DatasetCreation: Functions for creating a datset class with the previous downloaded data.
                   You will also need to download manually the zip folder "illumination_gt" and extract the files.
                  
- unet: it contains 3 different models. They are different variations of unet architecture.
- train2: it contrains the code for trainning using one of the models of the unet file.
- model and train: just ignore this files for now

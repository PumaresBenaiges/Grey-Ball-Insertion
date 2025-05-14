DEEP LEARNING PROJECT: 
GREY BALL INSERTION

Dataset:
- https://www2.cs.sfu.ca/~colour/data2/DRONE-Dataset/
  
Pyton files:
- **DataImport**: Downloads all the necessary images needed for the project into 2 new folders: scenes and scenes_shots.
- **DatasetCreation**: Functions for creating a datset class with the previous downloaded data.
                   You will also need to download manually the zip folder "illumination_gt" and extract the files.
- **Utils**: Set of functions used to load data or for the Dataset Creation.
                  
- **unet**: it contains 2 different models.
    - The unet: Features are computed at begging and passed to the model and concatenated in the bottleneck.
    - The unet with mobilnet for feature extraction, with the aim to train both.
- **train_unet**: it contrains the code for trainning the unet.
- **train_unet_mobilenet**: for trainning unet+mobilenetv3.
- **align_scenes**: functions to align the scenes with the precomputed transformations.

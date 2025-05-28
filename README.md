DEEP LEARNING PROJECT: 
GREY BALL INSERTION

Dataset:
- https://www2.cs.sfu.ca/~colour/data2/DRONE-Dataset/
  
Pyton files:

Preprocessing:
- **DataImport**: Downloads all the necessary images needed for the project into 2 new folders: scenes and scenes_shots.
  You will also need to download manually the zip folder "illumination_gt" and extract the files.

- **align_scenes**:  to apply H to all scenes and save transformed scenes if you have enough memory in computer. WE CAN ACTUALLY DELETE THIS FILE?
- **ball_segment_anything**: ?
- **yolo_find_center_ball_test.py**: ?

Model trainning:
- **DatasetCreation**: Functions for creating the dataset class (final approach).
- **DatasetCreation_previous**: Functions for creating a dataset class (first approach).
- **Utils**: Set of functions that are used in other files.
- **unet**: it contains 2 different models.
    - The unet: Features are computed at begging and passed to the model and concatenated in the bottleneck.
    - The unet with mobilnet for feature extraction, with the aim to train both.

- **run_model**: it contains the code for trainning unet+mobilenetv3 on the final approach.
- **train_unet**: it contrains the code for trainning the unet (mobilenet features are precomputed and stored).
- **train_unet_mobilenet**: for trainning unet+mobilenetv3.
- **print_res**: ? 


CSV files:
- **ball_data_modified.csv**: File obtained from running ?. It contains center coordinates and radious of the ball and they are used to crop the images.
- **homography_transformations_csv**: File obtained from running ?. It is used in order to transform the scenes so they match the position of scene shots.



DEEP LEARNING PROJECT: 
GREY BALL INSERTION

# 🚀 How to Run


## 1. Download the dataset
```bash
python DataImport.py
```
## 2. Preprocess the data using SAM to segment and crop the ball
```bash
python ball_segment_anything.py
```

## 3. Train the model (U-Net with MobileNetV3 as feature extractor)
```bash
python train_unet_mobilenet.py
```

## 4. View training logs using TensorBoard
```bash
python print_tensorboard_logs.py
```

## 5. Visualize model predictions on evaluation dataset
```bash
python overlay_model_results.py
```
# 📚 Dataset
- https://www2.cs.sfu.ca/~colour/data2/DRONE-Dataset/

## 🧠 U-Net + MobileNetV3 Model
- checkpoint_epoch_100.pt
  
# 📁 Project Structure


## Pyton files:

### Dataset diagnostic 
- **Image_check_ID**: logs the missing scene shots IDs in a file named Image_check_results.txt.
- **Transformation**: visually verify that the homography transformation correctly aligns the reference scene with each scene shot.

### Preprocessing:
- **DataImport**: Downloads all the necessary images needed for the project into 2 new folders: scenes and scenes_shots.
  You will also need to download manually the zip folder "illumination_gt" and extract the files.

- **ball_segment_anything**: use the Segment Anything Model (SAM) in order to:
- **Feature_Matching_fast**: compute and save to CSV fie geometric transformations between reference scenes and real-world shots

  - Crop and mask images to isolate the ball in each scene shot, then save these edited images in a separate folder. 

  -  Calculate the new center position of the ball and determine its radius. Save this information in a CSV file.

### Model trainning:
- **DatasetCreation**: Functions for creating the dataset class (final approach).
- **DatasetCreation_previous**: Functions for creating a dataset class (first approach).
- **Utils**: Set of functions that are used in other files.
- **unet**: it contains 2 different models.
    - The unet: Features are computed at begging and passed to the model and concatenated in the bottleneck.
    - The unet with mobilnet for feature extraction, with the aim to train both.

- **train_unet**: it contrains the code for trainning the unet (mobilenet features are precomputed and stored).
- **train_unet_mobilenet**: for trainning unet+mobilenetv3.

### Display results:
- **overlay_model_results**: overlay output of the model over the background image.
- **print_tensorboard_logs**: print tensorbard logs

## CSV files:
- **ball_data_modified.csv**: contains center coordinates and radious of the ball and they are used to crop the images.
- **homography_transformations_csv**: contains homography transformation matrixes to align scene with scene shots.



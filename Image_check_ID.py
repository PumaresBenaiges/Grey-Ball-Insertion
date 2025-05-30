"""
Sometimes there are images in the dataset that do not have corresponding coordinates in the "illumination_gt.csv", therefore we need to skip them for training.
To check how many scene shots are affected, run the code and look at the resulting "Image_check_results.txt"
"""

import os
import csv
import requests
import pandas as pd
from bs4 import BeautifulSoup

def get_ids_from_website(folder_url):
    """
    Find IDs in the internet by fetching wepsite content
    input: URL of website
    output: list with IDs in the corrseponding online folder
    """
    
    # Fetch the webpage content
    response = requests.get(folder_url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # List to store image IDs
    ids = []
    
    # Extract image names and filter those starting with "DSC_"
    for img in soup.find_all("a"):
        href = img.get("href")
        if href:
            filename = os.path.basename(href)  
            if filename.lower().startswith("dsc_"):
                if filename.lower().endswith(".nef"):
                    filename = filename.rsplit('.', 1)[0]
                    ids.append(filename)
    
    return ids

def get_ids_from_csv(file_path):
    """
    Find IDs from a csv
    input: file path of csv file
    output: list with IDs in the corrseponding csv file
    """
    
    # Load the CSV, skipping the first row
    df = pd.read_csv(f"{file_path}", sep=",") 
    # Get data from the first column
    ids = df.iloc[:, 0].tolist()
    # Remove ".jpg" extension from filenames
    ids = [img.rsplit('.', 1)[0] if img.lower().endswith(".jpg") else img for img in ids]
    
    return ids


def check_images_in_folders(folder_path, website_url, folders):
     """
     Function to check missing images in folders
     input: URL of Website, Path of csv file, name of scenes (folders)
     output: txt file that contains information if image in internet has corresponding coordinates in the csv file
     """

    # Define where to save txt file that contains the results
    result_filename = f"Image_check_results.txt"
    with open(result_filename, 'w') as result_file:

        # Iterate through each folder in the list
        for folder_name in folders:
            folder_url = f"{website_url}/{folder_name}"
            #print(f"Constructed folder URL: {folder_url}")

            # Get IDs from Internet
            image_ids_web = get_ids_from_website(folder_url)
            #print('Image id web:', image_ids_web)

            # Get IDs from csv file
            file_path = f"{folder_path}\{folder_name}.csv"
            print (file_path)

            image_ids_csv = get_ids_from_csv(file_path)
            #print('Image id csv:', image_ids_csv)
            
            result_file.write(f"Checking images for folder: {folder_name}\n")

            # Check if ID from internet is the same as the one in the csv file
            unique = list(set(image_ids_web) - set(image_ids_csv))
            print('Unique:', unique)
            result_file.write(f"{unique} has no coordinates.\n")
                
                
        result_file.write("\nCompleted checking images.\n")
        print(f"Finished")

# Sample usage
website_url = 'https://www2.cs.sfu.ca/~colour/data2/DRONE-Dataset/scenes_shots'  # Base URL of the website
folder_path = 'illumination_gt'  # Path to the folder containing CSV files
folders = ['A&W1', 'A&W2', 'A&W3', 'Harbour1', 'Harbour2', 'Harbour3', 'Harbour4', 'SFU_art', 
           'blue_ceiling', 'dining_area', 'downtown_smith', 'edu_area', 'foodcourt_mcnz', 
           'hallway', 'image_theater', 'owl_statue', 'playground', 'rugs', 'seat_rows', 
           'study_area', 'stump', 'subway1', 'subway2', 'theater', 'tree_tunel', 
           'uncle_faith1', 'uncle_faith2', 'under_tree2', 'wall_art', 'wall_hallway', 'wall_lab']

# Process all the folders and save the results in text files
check_images_in_folders(folder_path, website_url, folders)



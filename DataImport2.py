import os
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from io import BytesIO
from PIL import Image  # Optional, if you want to convert NEF to image (might need rawpy for NEF)

# Initialize an empty list to hold datasets in memory
# instead of downloading saved in a variable
scenes = []
shots = []

def download_file_to_dataset(file_url):
    try:
        # Send GET request
        file_response = requests.get(file_url, timeout=30)
        if file_response.status_code == 200:
            # Store content in memory
            image_bytes = file_response.content

            # Optional: Convert NEF raw image to an RGB image (requires rawpy or similar)
            # For now, we store raw bytes
            scene_id = os.path.splitext(file_url)[0]
            scenes.append({'url': scene_id, 'content': image_bytes})
            print(f"Downloaded and stored in dataset: {file_url}")
        else:
            print(f"Failed to download: {file_url} (Status code: {file_response.status_code})")
    except requests.RequestException as e:
        print(f"Error downloading {file_url}: {e}")

def download_scenes_to_dataset():
    base_url = 'https://www2.cs.sfu.ca/~colour/data2/DRONE-Dataset/scenes/'
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    file_links = [urljoin(base_url, link['href']) for link in soup.find_all('a', href=True)
                  if link['href'].lower().endswith('.nef')]

    # Download all scenes in parallel
    with ThreadPoolExecutor(max_workers=32) as executor:
        executor.map(download_file_to_dataset, file_links)

def download_scenes_with_ball_to_dataset():
    folders = ['A&W1', 'A&W2', 'A&W3', 'Harbour1', 'Harbour2', 'Harbour3',
               'Harbour4', 'SFU_art', 'blue_ceiling', 'dining_area',
               'downtown_smith', 'edu_area', 'foodcourt_mcnz', 'hallway',
               'image_theater', 'owl_statue', 'playground', 'rugs', 'seat_rows',
               'study_area', 'stump', 'subway1', 'subway2', 'theater', 'tree_tunel',
               'uncle_faith1', 'uncle_faith2', 'under_tree2', 'wall_art',
               'wall_hallway', 'wall_lab']

    for folder in folders:
        base_url = f'https://www2.cs.sfu.ca/~colour/data2/DRONE-Dataset/scenes_shots/{folder}/'
        response = requests.get(base_url)
        soup = BeautifulSoup(response.text, 'html.parser')

        file_links = [urljoin(base_url, link['href']) for link in soup.find_all('a', href=True)
                      if link['href'].lower().endswith('.nef')]

        with ThreadPoolExecutor(max_workers=64) as executor:
            executor.map(download_file_to_dataset, file_links)

        print(f"Downloaded all files in dataset for folder: {folder}")

if __name__ == '__main__':
    print('Download of scenes starting')
    download_scenes_to_dataset()
    print('Download of scenes finished')

    print('Download of scenes with ball starting')
    download_scenes_with_ball_to_dataset()
    print('Download of scenes with ball finished')

    print(f"Total images stored in dataset: {len(dataset)}")

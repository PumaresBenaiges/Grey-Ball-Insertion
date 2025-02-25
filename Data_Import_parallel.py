import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor

# Base URL of the folder
folder = 'downtown_smith'
base_url = 'https://www2.cs.sfu.ca/~colour/data2/DRONE-Dataset/scenes_shots/' + folder + '/'

# Send a GET request to the URL
response = requests.get(base_url)

# Parse the content using BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Find all anchor tags (<a>) that contain file links
file_links = soup.find_all('a', href=True)

# Create a directory to save files
os.makedirs(folder, exist_ok=True)

def download_file(link):
    # Get the file URL
    file_url = urljoin(base_url, link['href'])

    # Check if it's a file (not a subdirectory)
    if not file_url.endswith('/') and file_url.lower().endswith('.nef'):
        file_name = os.path.join(folder, link['href'])
        try:
            # Send a GET request to download the file
            file_response = requests.get(file_url, timeout=10)

            # Save the file to the directory if the request was successful
            if file_response.status_code == 200:
                with open(file_name, 'wb') as file:
                    file.write(file_response.content)
                print(f"Downloaded: {file_name}")
            else:
                print(f"Failed to download: {file_url} (Status code: {file_response.status_code})")

        except requests.RequestException as e:
            print(f"Error downloading {file_url}: {e}")

# Download files in parallel
max_workers = 32
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    executor.map(download_file, file_links)

print("All files have been downloaded.")

import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Base URL of the folder
base_url = 'https://www2.cs.sfu.ca/~colour/data2/DRONE-Dataset/scenes_shots/A&W1/'

# Send a GET request to the URL
response = requests.get(base_url)

# Parse the content using BeautifulSoup
soup = BeautifulSoup(response.text, 'html.parser')

# Find all anchor tags (<a>) that contain file links
file_links = soup.find_all('a', href=True)

# Create a directory to save files
os.makedirs('downloaded_files', exist_ok=True)

# Loop over each link and download the .nef files only
for link in file_links:
    # Get the file URL
    file_url = urljoin(base_url, link['href'])

    # Check if it's a file (not a subdirectory) and if it's a .nef file
    if not file_url.endswith('/') and file_url.endswith('.NEF'):
        file_name = os.path.join('downloaded_files', link['href'])

        # Send a GET request to download the file
        file_response = requests.get(file_url)

        # Save the file to the directory
        with open(file_name, 'wb') as file:
            file.write(file_response.content)

        print(f"Downloaded: {file_name}")

print("All .nef files have been downloaded.")

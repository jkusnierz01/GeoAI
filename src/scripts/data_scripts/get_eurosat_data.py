import requests
import zipfile
import os
from tqdm import tqdm

def download_eurosat_ms(target_folder="."):
    url = "https://zenodo.org/record/7711810/files/EuroSAT_MS.zip"
    filename = "EuroSAT_MS.zip"
    file_path = os.path.join(target_folder, filename)

    # 1. Download the file with a progress bar
    print(f"Downloading {filename} from Zenodo...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte

    with open(file_path, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)

    # 2. Extract the contents
    print("\nDownload complete. Extracting files...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)
    
    print(f"Success! Dataset extracted to: {os.path.abspath(target_folder)}")

if __name__ == "__main__":
    download_eurosat_ms()
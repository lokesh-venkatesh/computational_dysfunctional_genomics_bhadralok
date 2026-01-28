import os
import requests
import gzip
import shutil

folder_name = "hg38"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

base_url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes"

# Download and extract chromosome files
for chrom in range(1, 23):
    filename = f"chr{chrom}.fa.gz"
    url = f"{base_url}/{filename}"
    filepath = os.path.join(folder_name, filename)
    
    print(f"Downloading {filename}...")
    response = requests.get(url)
    with open(filepath, 'wb') as f:
        f.write(response.content)
    
    # Unzip the file
    print(f"Extracting {filename}...")
    extracted_path = os.path.join(folder_name, filename[:-3])  # Remove .gz extension
    with gzip.open(filepath, 'rb') as f_in:
        with open(extracted_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Remove the gz file
    os.remove(filepath)

print("All files downloaded and extracted successfully!")
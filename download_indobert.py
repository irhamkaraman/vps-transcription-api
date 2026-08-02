import gdown
import os

url = 'https://drive.google.com/drive/folders/1pvYfI8jA41MN3H4XIvnNFw6EmhPglAiy?usp=drive_link'
output_dir = 'models'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Downloading folder from Google Drive to {output_dir}...")
gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)
print("Download complete.")

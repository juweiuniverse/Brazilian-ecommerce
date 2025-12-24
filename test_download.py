"""Simple test to validate Kaggle authentication and download (non-destructive).

Usage:
  Set environment variables KAGGLE_USERNAME and KAGGLE_KEY (or use .streamlit/secrets.toml for local testing),
  then run: python test_download.py

The script will try to authenticate and check that the dataset 'olistbr/brazilian-ecommerce' is accessible.
"""
import os
import json
from pathlib import Path
import sys

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except Exception as e:
    print('kaggle package not installed:', e)
    sys.exit(2)

KG_DIR = Path(os.environ.get('KAGGLE_CONFIG_DIR', Path.cwd()))
KG_DIR.mkdir(parents=True, exist_ok=True)

username = os.environ.get('KAGGLE_USERNAME')
key = os.environ.get('KAGGLE_KEY')
if username and key:
    kaggle_json = KG_DIR / 'kaggle.json'
    with open(kaggle_json, 'w') as f:
        json.dump({'username': username, 'key': key}, f)
    print('Wrote kaggle.json to', kaggle_json)
else:
    print('No KAGGLE_USERNAME / KAGGLE_KEY found in environment. If you use .streamlit/secrets.toml, load it or set env vars.')

os.environ['KAGGLE_CONFIG_DIR'] = str(KG_DIR)

api = KaggleApi()
try:
    api.authenticate()
    print('Authentication OK')
    # List files in dataset metadata (does not download)
    try:
        files = api.dataset_list_files('olistbr/brazilian-ecommerce')
        print('Dataset files count:', len(files.files))
        for f in files.files[:10]:
            print('-', f.name)
    except Exception as e:
        print('Could not list dataset files:', e)
except Exception as e:
    print('Authentication failed:', e)
    sys.exit(1)

print('Done')

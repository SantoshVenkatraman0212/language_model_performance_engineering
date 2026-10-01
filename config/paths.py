'''
This file has the file & dir paths used throughout this repo
'''
# Importing necessary libraries
from pathlib import Path

# Getting parent dir i.e. main repo path
ROOT_DIR = Path(__file__).resolve().parent.parent
ATTN_COMPARISON_IMG_DIR = f'{ROOT_DIR}/tensor_computation/attention_comparison_images'
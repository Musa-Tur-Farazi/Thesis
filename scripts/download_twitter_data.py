#!/usr/bin/env python3
"""
download_twitter_data.py - Download and process Twitter dataset from UCR STAR portal
"""

import os
import sys
import requests
import json
import pandas as pd
from tqdm import tqdm

# Add parent directory to path to ensure we use the local package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# URL for the Twitter dataset from STAR portal
# The dataset can be downloaded in GeoJSON format
TWITTER_DATASET_URL = "https://star.cs.ucr.edu/dynamic/download/Tweets.geojson"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "twitter_data.json")

def download_twitter_data():
    """
    Download the Twitter dataset from UCR STAR portal if it doesn't exist locally
    """
    if os.path.exists(OUTPUT_FILE):
        print(f"Twitter dataset already exists at {OUTPUT_FILE}")
        return OUTPUT_FILE
    
    print(f"Downloading Twitter dataset from {TWITTER_DATASET_URL}...")
    try:
        response = requests.get(TWITTER_DATASET_URL, stream=True)
        response.raise_for_status()
        
        # Get the total file size
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        # Use tqdm to show download progress
        with open(OUTPUT_FILE, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
            for data in response.iter_content(block_size):
                f.write(data)
                progress_bar.update(len(data))
        
        print(f"Dataset downloaded to {OUTPUT_FILE}")
        return OUTPUT_FILE
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        print("Please manually download the dataset from the STAR portal:")
        print("1. Visit https://star.cs.ucr.edu/?Tweets")
        print("2. Click on 'Download data' and select GeoJSON format")
        print(f"3. Save the file as {OUTPUT_FILE}")
        return None

def load_twitter_data(max_points=None):
    """
    Load the Twitter dataset and extract points
    
    Args:
        max_points: Maximum number of points to load (None for all)
        
    Returns:
        List of (x, y) coordinates
    """
    # Make sure the data is downloaded
    if not os.path.exists(OUTPUT_FILE):
        file_path = download_twitter_data()
        if not file_path:
            raise ValueError("Dataset not available. Please download manually.")
    else:
        file_path = OUTPUT_FILE
    
    print(f"Loading Twitter dataset from {file_path}...")
    
    # Try different encodings to handle potential encoding issues
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    data = None
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                try:
                    data = json.load(f)
                    print(f"Successfully loaded JSON data using {encoding} encoding")
                    break
                except json.JSONDecodeError:
                    # Not a valid JSON with this encoding, try next one or CSV format
                    continue
        except UnicodeDecodeError:
            continue
    
    # If we couldn't load as JSON with any encoding, try CSV
    if data is None:
        for encoding in encodings:
            try:
                # Try to handle as CSV
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"Successfully loaded CSV data using {encoding} encoding")
                    
                    # Convert to points - assumes latitude and longitude columns
                    if 'latitude' in df.columns and 'longitude' in df.columns:
                        points = [(row['longitude'], row['latitude']) for _, row in df.iterrows()]
                    else:
                        # Try to find any columns that might contain coordinates
                        lat_col = None
                        lon_col = None
                        for col in df.columns:
                            if 'lat' in col.lower():
                                lat_col = col
                            if 'lon' in col.lower() or 'lng' in col.lower():
                                lon_col = col
                        
                        if lat_col and lon_col:
                            points = [(row[lon_col], row[lat_col]) for _, row in df.iterrows()]
                        else:
                            raise ValueError("Could not identify latitude/longitude columns")
                    
                    if max_points:
                        points = points[:max_points]
                    
                    return points
                except Exception as e:
                    print(f"Error parsing as CSV with {encoding} encoding: {str(e)}")
                    continue
            except UnicodeDecodeError:
                continue
    
    # If we couldn't load the file as either JSON or CSV, try binary mode
    if data is None:
        try:
            with open(file_path, 'rb') as f:
                # Try to decode as UTF-8 with error handling
                content = f.read().decode('utf-8', errors='replace')
                try:
                    data = json.loads(content)
                    print("Successfully loaded JSON data in binary mode with UTF-8 replacement")
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON in binary mode: {str(e)}")
                    raise ValueError("Could not parse dataset as JSON or CSV after trying multiple encodings")
        except Exception as e:
            print(f"Error reading file in binary mode: {str(e)}")
            raise ValueError("Could not read the dataset file with any supported encoding")
    
    # Extract points from GeoJSON
    points = []
    if 'features' in data:
        for feature in data['features']:
            if feature['geometry']['type'] == 'Point':
                # GeoJSON format has [longitude, latitude]
                coords = feature['geometry']['coordinates']
                points.append((coords[0], coords[1]))
    
    if not points:
        raise ValueError("No point data found in the dataset")
    
    if max_points:
        points = points[:max_points]
    
    print(f"Loaded {len(points)} points from Twitter dataset")
    return points

if __name__ == "__main__":
    # Download and load the data
    try:
        points = load_twitter_data(max_points=1000)  # Limit to 10,000 points for testing
        print(f"Successfully loaded {len(points)} points")
        print(f"Sample points: {points[:5]}")
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please download the dataset manually") 
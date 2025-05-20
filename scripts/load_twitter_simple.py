#!/usr/bin/env python3
"""
load_twitter_simple.py - A simple, robust parser for Twitter GeoJSON data that avoids encoding issues
"""

import os
import re
import sys
import random

# Path to the Twitter data file
DATA_FILE = os.path.join(os.path.dirname(__file__), "twitter_data.json")

def extract_coordinates(max_points=None):
    """
    Extract coordinates from Twitter data using regex to avoid encoding issues.
    This function reads the file line by line and extracts coordinates using regex.
    
    Args:
        max_points: Maximum number of points to extract (None for all)
        
    Returns:
        List of (x, y) coordinate tuples
    """
    if not os.path.exists(DATA_FILE):
        print(f"Error: Twitter data file not found at {DATA_FILE}")
        return []
    
    points = []
    # Regular expression to match coordinates in "coordinates": [x, y] format
    coord_pattern = re.compile(r'"coordinates":\s*\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]')
    
    try:
        # Open in binary mode to avoid encoding issues
        with open(DATA_FILE, 'rb') as f:
            line_count = 0
            for raw_line in f:
                line_count += 1
                try:
                    # Decode with errors='replace' to handle problematic characters
                    line = raw_line.decode('utf-8', errors='replace')
                    
                    # Find all coordinates in the line
                    matches = coord_pattern.findall(line)
                    for match in matches:
                        try:
                            x, y = float(match[0]), float(match[1])
                            points.append((x, y))
                        except ValueError:
                            # Skip invalid coordinates
                            continue
                            
                    if max_points and len(points) >= max_points:
                        break
                        
                except Exception as e:
                    print(f"Error processing line {line_count}: {str(e)}")
                    continue
    except Exception as e:
        print(f"Error reading Twitter data file: {str(e)}")
    
    print(f"Extracted {len(points)} points from Twitter data using simple parser")
    
    # Shuffle points to avoid any ordering bias if taking a subset
    random.shuffle(points)
    
    if max_points and len(points) > max_points:
        points = points[:max_points]
        
    return points

if __name__ == "__main__":
    points = extract_coordinates(max_points=100)
    if points:
        print(f"Successfully extracted {len(points)} points")
        print(f"Sample points: {points[:5]}")
    else:
        print("No points could be extracted from the file.") 
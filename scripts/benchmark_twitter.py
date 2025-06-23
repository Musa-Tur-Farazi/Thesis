#!/usr/bin/env python3
"""
benchmark_twitter.py - Benchmark R-tree implementations using Twitter data
"""

import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Any

# Add parent directory to path to ensure we use the local package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtreelib import Rect, RTree, RStarTree, BasicNNRTree
from download_twitter_data import load_twitter_data

# Try to import the simple parser - it's okay if it fails
try:
    from load_twitter_simple import extract_coordinates
    have_simple_parser = True
except ImportError:
    have_simple_parser = False

# Constants for the experiment
CAPACITY = 16  # Maximum capacity of R-tree nodes
MAX_POINTS = 100  # Maximum number of points to use from Twitter dataset

# Toggle this flag to include or exclude the (heavier) Fisher-Flow R-Tree benchmark.
INCLUDE_FFM = False

def normalize_points(points):
    """
    Normalize points to a reasonable coordinate space for the experiment
    """
    # Extract x and y coordinates
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    # Calculate min/max for normalization
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Normalize to [-1000, 1000] range
    x_range = max_x - min_x
    y_range = max_y - min_y
    
    normalized = []
    for x, y in points:
        norm_x = -1000 + ((x - min_x) / x_range) * 2000
        norm_y = -1000 + ((y - min_y) / y_range) * 2000
        normalized.append((norm_x, norm_y))
    
    print(f"Normalized coordinates from [{min_x}, {max_x}] x [{min_y}, {max_y}] to [-1000, 1000] x [-1000, 1000]")
    return normalized

def filter_valid_points(points):
    """
    Filter out invalid or missing coordinates
    """
    return [(x, y) for x, y in points if not (np.isnan(x) or np.isnan(y))]

def sample_rects(points, n, avg_width_perc, avg_height_perc):
    """
    Generate n random rectangles based on the distribution of points
    """
    if not points:
        return []
    
    # Extract x and y coordinates
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    # Calculate bounds
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x_range = max_x - min_x
    y_range = max_y - min_y
    
    # Calculate avg width and height in coordinate units
    avg_width = x_range * avg_width_perc
    avg_height = y_range * avg_height_perc
    
    # Generate rectangle centers near actual data points
    indices = np.random.choice(len(points), min(n, len(points)), replace=False)
    centers = [points[i] for i in indices]
    
    # Generate rectangles with varied sizes
    rects = []
    for cx, cy in centers:
        # Vary the width and height
        width = avg_width * np.random.uniform(0.5, 1.5)
        height = avg_height * np.random.uniform(0.5, 1.5)
        
        # Create rectangle
        rect = (cx - width/2, cy - height/2, cx + width/2, cy + height/2)
        rects.append(rect)
    
    return rects

def build_rtree(pts, rtree_class, **kwargs):
    """Build an R-tree from points using the specified implementation."""
    print(f"Building {rtree_class.__name__}...")
    t0 = time.time()
    tree = rtree_class(**kwargs)
    
    # Use a small epsilon to ensure point rectangles have a minimal size
    epsilon = 0.0001
    
    for i, (x, y) in enumerate(pts):
        # Instead of zero-area rectangles, use tiny rectangles
        tree.insert(i, Rect(x-epsilon, y-epsilon, x+epsilon, y+epsilon))
    build_time = time.time() - t0
    return tree, build_time * 1000

def query_and_measure(tree, rects):
    """Query the tree and measure performance."""
    t0 = time.time()
    results = []
    hit_counts = []
    
    # Reset node access counter
    tree.reset_node_accesses()
    
    for xmin, ymin, xmax, ymax in rects:
        query_rect = Rect(xmin, ymin, xmax, ymax)
        query_results = list(tree.query(query_rect))
        results.append(query_results)
        hit_counts.append(len(query_results))
    
    query_time = (time.time() - t0) / len(rects) * 1000
    
    # Return node access count along with other metrics
    return results, query_time, hit_counts, tree.node_accesses

def visualize_rtree(tree, pts, rect, results, title=None):
    """Visualize an R-tree structure with query rectangle and results."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    
    # Plot all points
    xs, ys = zip(*pts)
    ax.scatter(xs, ys, s=4, c='lightgrey', alpha=0.5)
    
    # Draw query rectangle
    xmin, ymin, xmax, ymax = rect
    width = xmax - xmin
    height = ymax - ymin
    
    # Zoom in on the query rectangle with some padding
    padding = max(width, height) * 2
    ax.set_xlim(xmin - padding, xmax + padding)
    ax.set_ylim(ymin - padding, ymax + padding)
    
    ax.add_patch(patches.Rectangle((xmin, ymin), 
                               width, 
                               height,
                               ec='red', 
                               fc='none', 
                               lw=2))
    
    # Highlight results
    result_ids = [e.data for e in results]
    result_pts = [pts[i] for i in result_ids if i < len(pts)]
    if result_pts:
        rx, ry = zip(*result_pts)
        ax.scatter(rx, ry, s=80, c='blue', marker='x', linewidths=2)
        
        # Add a larger red circle around each result point to make them stand out
        ax.scatter(rx, ry, s=200, facecolors='none', edgecolors='red', linewidths=1)
    
    # Set title and display
    if title:
        if results:
            title = f"{title} ({len(results)} results)"
        ax.set_title(title)
    
    return fig, ax

def visualize_rtree_structure(tree, level=1):
    """Visualize the structure of an R-tree at a specific level."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    
    levels = tree.get_levels()
    if level >= len(levels):
        print(f"Warning: Level {level} does not exist. Max level is {len(levels)-1}")
        level = len(levels) - 1
    
    # Draw rectangles for the specified level
    for node in levels[level]:
        rect = node.get_bounding_rect()
        xmin, ymin = rect.min_x, rect.min_y
        xmax, ymax = rect.max_x, rect.max_y
        ax.add_patch(patches.Rectangle(
            (xmin, ymin), 
            xmax - xmin, 
            ymax - ymin,
            ec='blue', 
            fc='blue', 
            alpha=0.2,
            lw=1
        ))
    
    ax.set_title(f"R-tree Structure (Level {level})")
    
    return fig, ax

def compare_rtree_structures(trees, tree_names, level=1):
    """Compare the structure of multiple R-trees at a specific level."""
    fig, axes = plt.subplots(1, len(trees), figsize=(6*len(trees), 6))
    if len(trees) == 1:
        axes = [axes]
    
    for ax, tree, name in zip(axes, trees, tree_names):
        ax.set_aspect('equal')
        
        levels = tree.get_levels()
        if level >= len(levels):
            print(f"Warning: Level {level} does not exist in {name}. Max level is {len(levels)-1}")
            level = len(levels) - 1
        
        # Draw rectangles for the specified level
        for node in levels[level]:
            rect = node.get_bounding_rect()
            xmin, ymin = rect.min_x, rect.min_y
            xmax, ymax = rect.max_x, rect.max_y
            ax.add_patch(patches.Rectangle(
                (xmin, ymin), 
                xmax - xmin, 
                ymax - ymin,
                ec='blue', 
                fc='blue', 
                alpha=0.2,
                lw=1
            ))
        
        ax.set_title(f"{name} (Level {level})")
    
    plt.tight_layout()
    return fig, axes

def evaluate_accuracy(tree_results, reference_results):
    """Calculate precision, recall, and F1 score."""
    tp = fn = fp = 0
    
    for tree_res, ref_res in zip(tree_results, reference_results):
        tree_ids = set(e.data for e in tree_res)
        ref_ids = set(e.data for e in ref_res)
        
        # Calculate TP, FP, FN
        query_tp = len(tree_ids & ref_ids)
        query_fp = len(tree_ids - ref_ids)
        query_fn = len(ref_ids - tree_ids)
        
        tp += query_tp
        fp += query_fp
        fn += query_fn
    
    # Calculate metrics
    prec = 1.0 if tp + fp == 0 else tp / (tp + fp)
    rec = 1.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    
    # Print detailed metrics
    print(f"Accuracy metrics: TP={tp}, FP={fp}, FN={fn}")
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    return prec, rec, f1

def create_synthetic_twitter_data(num_points=10000):
    """
    Create synthetic data similar to Twitter data as a fallback if loading fails.
    This generates points with higher density in certain areas (like cities).
    
    Args:
        num_points: Number of points to generate
        
    Returns:
        List of (x, y) coordinate tuples
    """
    print("Creating synthetic Twitter-like data as fallback...")
    # Create a mixture of clusters to simulate population centers
    np.random.seed(42)  # For reproducibility
    
    # Generate cluster centers to simulate cities
    num_clusters = 15
    cluster_centers = []
    for _ in range(num_clusters):
        # Random coordinates for cluster centers
        center_x = np.random.uniform(-90, -70)  # Roughly eastern US longitude
        center_y = np.random.uniform(25, 45)   # Roughly US latitude
        cluster_centers.append((center_x, center_y))
    
    # Generate points around these clusters
    points = []
    points_per_cluster = num_points // (num_clusters + 1)  # Leave some for random noise
    
    # Generate cluster points
    for center_x, center_y in cluster_centers:
        # Standard deviation controls cluster size (smaller = tighter cluster)
        std_dev = np.random.uniform(0.5, 2.0)  
        
        # Generate points with normal distribution around center
        for _ in range(points_per_cluster):
            x = np.random.normal(center_x, std_dev)
            y = np.random.normal(center_y, std_dev)
            points.append((x, y))
    
    # Add some random noise points
    remaining = num_points - len(points)
    for _ in range(remaining):
        x = np.random.uniform(-95, -65)  # Wider range for noise
        y = np.random.uniform(20, 50)
        points.append((x, y))
    
    print(f"Created {len(points)} synthetic points")
    return points

def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Try loading Twitter dataset with various methods
    pts = None
    
    # Method 1: Try the regular parser first
    try:
        raw_points = load_twitter_data(max_points=MAX_POINTS)
        valid_points = filter_valid_points(raw_points)
        print(f"Filtered {len(raw_points)} points to {len(valid_points)} valid points")
        
        # Make sure we got enough points
        if len(valid_points) >= 5:  # Need at least some reasonable number of points
            # Normalize the points to a reasonable coordinate space
            pts = normalize_points(valid_points)
            
            # Display some sample points
            print(f"Sample points after normalization: {pts[:5]}")
    except Exception as e:
        print(f"Error loading Twitter data with primary method: {str(e)}")
    
    # Method 2: If first method failed, try the simple parser
    if pts is None and have_simple_parser:
        try:
            print("Trying simple parser method...")
            raw_points = extract_coordinates(max_points=MAX_POINTS)
            if raw_points:
                valid_points = filter_valid_points(raw_points)
                print(f"Simple parser found {len(valid_points)} valid points")
                
                if len(valid_points) >= 100:
                    # Normalize the points to a reasonable coordinate space
                    pts = normalize_points(valid_points)
                    
                    # Display some sample points
                    print(f"Sample points after normalization (simple parser): {pts[:5]}")
        except Exception as e:
            print(f"Error using simple parser: {str(e)}")
    
    # Method 3: If both real data methods failed, use synthetic data
    if pts is None:
        print("Falling back to synthetic Twitter-like data...")
        # Use synthetic data instead
        raw_points = create_synthetic_twitter_data(MAX_POINTS)
        valid_points = filter_valid_points(raw_points)
        print(f"Created {len(valid_points)} synthetic points")
        
        # Normalize the points to a reasonable coordinate space
        pts = normalize_points(valid_points)
        
        # Display some sample points
        print(f"Sample synthetic points after normalization: {pts[:5]}")
    
    # Generate queries of different sizes
    # Use percentages of the total coordinate space to define query sizes
    small_queries = sample_rects(pts, 1, 0.01, 0.01)  # Small (1% of coordinate space)
    medium_queries = sample_rects(pts, 1, 0.05, 0.05)  # Medium (5% of coordinate space)
    large_queries = sample_rects(pts, 1, .2, .2)  # Large (20% of coordinate space)
    
    # Combine all queries
    all_queries = small_queries + medium_queries + large_queries
    print(f"Generated {len(all_queries)} queries of varying sizes")
    
    # Print all data points for manual verification
    print("\n===== ALL DATA POINTS =====")
    for i, (x, y) in enumerate(pts):
        print(f"Point {i}: ({x:.6f}, {y:.6f})")
        
    # Print all query rectangles
    print("\n===== ALL QUERY RECTANGLES =====")
    for i, (xmin, ymin, xmax, ymax) in enumerate(all_queries):
        print(f"Query {i}: ({xmin:.6f}, {ymin:.6f}, {xmax:.6f}, {ymax:.6f})")
        # Calculate which points should be in this rectangle
        matching_pts = []
        for j, (x, y) in enumerate(pts):
            if xmin <= x <= xmax and ymin <= y <= ymax:
                matching_pts.append(j)
        print(f"  Expected matches: {matching_pts}")
    
    # Build trees
    print("\nBuilding R-trees...")
    rtree, rtree_time = build_rtree(pts, RTree, max_entries=CAPACITY)
    rstar, rstar_time = build_rtree(pts, RStarTree, max_entries=CAPACITY)
    
    # Fisher-Flow benchmark disabled; build only BasicNN-augmented tree
    bnn, bnn_time = build_rtree(pts, BasicNNRTree,
                                max_entries=CAPACITY,
                                min_train_samples=100,
                                router_hidden_size=32,
                                router_depth=1,
                                router_epochs=10,
                                router_top_k=1)
    
    print(f"Build times (ms): Classic={rtree_time:.2f}, R*={rstar_time:.2f}, BasicNN={bnn_time:.2f}")
    
    # Query trees
    print("\nQuerying trees...")
    rtree_results, rtree_query_time, rtree_hits, rtree_node_accesses = query_and_measure(rtree, all_queries)
    rstar_results, rstar_query_time, rstar_hits, rstar_node_accesses = query_and_measure(rstar, all_queries)
    bnn_results, bnn_query_time, bnn_hits, bnn_node_accesses = query_and_measure(bnn, all_queries)
    
    print(f"Query times (ms/query): Classic={rtree_query_time:.2f}, R*={rstar_query_time:.2f}, BasicNN={bnn_query_time:.2f}")
    
    # Print detailed query results for each tree implementation
    print("\n===== DETAILED QUERY RESULTS =====")
    for i, ((xmin, ymin, xmax, ymax), rtree_res, rstar_res, bnn_res) in enumerate(zip(all_queries, rtree_results, rstar_results, bnn_results)):
        print(f"\nQuery {i}: ({xmin:.6f}, {ymin:.6f}, {xmax:.6f}, {ymax:.6f})")
        
        # Print the result IDs from each implementation
        rtree_ids = sorted([e.data for e in rtree_res])
        rstar_ids = sorted([e.data for e in rstar_res])
        bnn_ids = sorted([e.data for e in bnn_res])
        
        print(f"  Classic R-tree results: {rtree_ids}")
        print(f"  R*-tree results: {rstar_ids}")
        print(f"  BasicNN-tree results: {bnn_ids}")
        
        # Calculate differences between implementations
        if rtree_ids != rstar_ids:
            rstar_missing = set(rtree_ids) - set(rstar_ids)
            rstar_extra = set(rstar_ids) - set(rtree_ids)
            print(f"  R* differences: missing={rstar_missing}, extra={rstar_extra}")
        
        if rtree_ids != bnn_ids:
            bnn_missing = set(rtree_ids) - set(bnn_ids)
            bnn_extra = set(bnn_ids) - set(rtree_ids)
            print(f"  BasicNN differences: missing={bnn_missing}, extra={bnn_extra}")
    
    # Calculate query hit statistics
    rtree_total_hits = sum(rtree_hits)
    rstar_total_hits = sum(rstar_hits)
    bnn_total_hits = sum(bnn_hits)
    
    print(f"\nTotal query hits: Classic={rtree_total_hits}, R*={rstar_total_hits}, BasicNN={bnn_total_hits}")
    
    # Report node access counts and pruning efficiency
    print(f"\nNode accesses: Classic={rtree_node_accesses}, R*={rstar_node_accesses}, BasicNN={bnn_node_accesses}")
    
    # Calculate pruning efficiency (% of nodes pruned compared to Guttman)
    total_nodes = len(list(rtree.get_nodes()))
    rstar_pruning = (1 - rstar_node_accesses / rtree_node_accesses) * 100
    bnn_pruning = (1 - bnn_node_accesses / rtree_node_accesses) * 100
    
    print(f"Pruning efficiency: R*={rstar_pruning:.2f}%, BasicNN={bnn_pruning:.2f}%")
    print(f"Total tree nodes: {total_nodes}")
    
    # Show hits per query type
    n_small = len(small_queries)
    n_medium = len(medium_queries)
    
    # Calculate hits by query type
    small_hits = sum(rtree_hits[:n_small])
    medium_hits = sum(rtree_hits[n_small:n_small+n_medium])
    large_hits = sum(rtree_hits[n_small+n_medium:])
    
    print(f"Hits by query type (Classic R-tree):")
    print(f"- Small queries: {small_hits} total, {small_hits/n_small:.2f} avg hits per query")
    print(f"- Medium queries: {medium_hits} total, {medium_hits/n_medium:.2f} avg hits per query")
    print(f"- Large queries: {large_hits} total, {large_hits/len(large_queries):.2f} avg hits per query")
    
    # Evaluate accuracy (using Guttman as reference)
    print("\nEvaluating accuracy against Classic R-tree:")
    print("R*-Tree vs Classic:")
    rstar_prec, rstar_rec, rstar_f1 = evaluate_accuracy(rstar_results, rtree_results)
    
    print("\nBasicNN-Tree vs Classic:")
    bnn_prec, bnn_rec, bnn_f1 = evaluate_accuracy(bnn_results, rtree_results)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Compare tree structures
    fig, _ = compare_rtree_structures(
        [rtree, rstar, bnn],
        ["Guttman", "R*", "BasicNN"],
        level=1
    )
    fig.savefig("twitter_rtree_structure_comparison_level1.png")
    print("Saved structure visualization: twitter_rtree_structure_comparison_level1.png")
    
    # Find a query with results to visualize
    results_found = False
    for i, rect in enumerate(all_queries):
        if rtree_hits[i] > 0:
            print(f"Found query {i} with {rtree_hits[i]} hits for visualization")
            rect_obj = Rect(*rect)
            
            # Query each implementation for this rectangle
            rtree_res = list(rtree.query(rect_obj))
            rstar_res = list(rstar.query(rect_obj))
            bnn_res = list(bnn.query(rect_obj))
            
            # Visualize query results
            visualize_rtree(rtree, pts, rect, rtree_res, "Guttman RTree Query")
            plt.savefig("twitter_guttman_query.png")
            
            visualize_rtree(rstar, pts, rect, rstar_res, "R* Tree Query")
            plt.savefig("twitter_rstar_query.png")
            
            visualize_rtree(bnn, pts, rect, bnn_res, "BasicNN RTree Query")
            plt.savefig("twitter_bnn_query.png")
            
            print("Saved query visualizations")
            results_found = True
            break
    
    if not results_found:
        print("No queries with results found for visualization")
    
    # Print summary
    print("\n===== TWITTER DATASET BENCHMARK SUMMARY =====")
    print(f"Dataset: Twitter data with {len(pts)} points")
    print(f"Queries: {len(all_queries)} total ({len(small_queries)} small, {len(medium_queries)} medium, {len(large_queries)} large)")
    
    print("\nBuild Time (ms):")
    print(f"{'RTree (Guttman)':20} {rtree_time:.2f}")
    print(f"{'RStarTree':20} {rstar_time:.2f}")
    print(f"{'BasicNNRTree':20} {bnn_time:.2f}")
    
    print("\nQuery Time (ms/query):")
    print(f"{'RTree (Guttman)':20} {rtree_query_time:.2f}")
    print(f"{'RStarTree':20} {rstar_query_time:.2f}")
    print(f"{'BasicNNRTree':20} {bnn_query_time:.2f}")
    
    print("\nNode Accesses:")
    print(f"{'RTree (Guttman)':20} {rtree_node_accesses}")
    print(f"{'RStarTree':20} {rstar_node_accesses}")
    print(f"{'BasicNNRTree':20} {bnn_node_accesses}")
    
    print("\nPruning Efficiency (% nodes pruned vs Guttman):")
    print(f"{'RTree (Guttman)':20} 0.00% (reference)")
    print(f"{'RStarTree':20} {rstar_pruning:.2f}%")
    print(f"{'BasicNNRTree':20} {bnn_pruning:.2f}%")
    
    print("\nAccuracy (vs. Guttman RTree):")
    print(f"{'RTree (Guttman)':20} precision=100.00% recall=100.00% F1=100.00% (reference)")
    print(f"{'RStarTree':20} precision={rstar_prec:.2%} recall={rstar_rec:.2%} F1={rstar_f1:.2%}")
    print(f"{'BasicNNRTree':20} precision={bnn_prec:.2%} recall={bnn_rec:.2%} F1={bnn_f1:.2%}")
    
    print("\nTwitter benchmark complete!")

if __name__ == "__main__":
    main() 
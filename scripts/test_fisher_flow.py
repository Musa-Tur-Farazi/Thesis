#!/usr/bin/env python3
"""
test_fisher_flow.py - Test the Fisher-Flow R-tree implementation
"""

import sys
import os
import random
import numpy as np
import time
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add parent directory to path to ensure we use the local package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtreelib import Rect, RTree, RStarTree, FisherFlowRTree

# Constants for data generation
XMIN, XMAX, YMIN, YMAX = -1000, 1000, -1000, 1000
CAPACITY = 64  # Maximum capacity of R-tree nodes

# Data generation functions
def sample_mixed(n_each=2500):
    """Generate a mixed dataset with uniform, Gaussian mixture, skewed, and Zipf distributions."""
    def uni(n):
        return np.c_[np.random.uniform(XMIN, XMAX, n),
                     np.random.uniform(YMIN, YMAX, n)]
    
    def gmm(n, k=10):
        from sklearn.mixture import GaussianMixture
        means = np.c_[np.random.uniform(XMIN, XMAX, k),
                      np.random.uniform(YMIN, YMAX, k)]
        gm = GaussianMixture(k, covariance_type="full", random_state=0)
        gm.fit(means)
        return gm.sample(n)[0]
    
    def skew(n, a=2, b=5):
        from scipy.stats import beta
        u, v = beta(a, b).rvs(n), beta(a, b).rvs(n)
        return np.c_[XMIN + u * (XMAX - XMIN), YMIN + v * (YMAX - YMIN)]
    
    def zipf(n, a=3):
        r = np.random.zipf(a, n)
        u, v = (r % 512) / 511, ((r // 512) % 512) / 511
        return np.c_[XMIN + u * (XMAX - XMIN), YMIN + v * (YMAX - YMIN)]
    
    return np.vstack([uni(n_each), gmm(n_each), skew(n_each), zipf(n_each)])

def sample_rects(n, avg_width, avg_height):
    """
    Generate n random rectangles with varied sizes around the specified average width and height.
    """
    cx = np.random.uniform(XMIN, XMAX, n)
    cy = np.random.uniform(YMIN, YMAX, n)
    
    # Vary the width and height to create queries of different sizes
    widths = np.random.uniform(avg_width * 0.2, avg_width * 1.8, n)
    heights = np.random.uniform(avg_height * 0.2, avg_height * 1.8, n)
    
    return [(x - w/2, y - h/2, x + w/2, y + h/2) 
            for x, y, w, h in zip(cx, cy, widths, heights)]

def sample_data_based_rects(pts, n, avg_width, avg_height):
    """
    Generate n random rectangles centered around actual data points.
    This ensures the queries will likely return results.
    """
    if len(pts) == 0:
        return []
    
    # Randomly select points to build rectangles around
    indices = np.random.choice(len(pts), min(n, len(pts)), replace=False)
    selected_pts = [pts[i] for i in indices]
    
    # Ensure minimum rectangle size to guarantee point inclusion
    # Use at least 1.0 for width/height to ensure we capture the point
    min_size = 1.0
    
    # Vary the width and height to create queries of different sizes
    widths = np.random.uniform(max(avg_width * 0.5, min_size), 
                              avg_width * 1.5, len(selected_pts))
    heights = np.random.uniform(max(avg_height * 0.5, min_size), 
                              avg_height * 1.5, len(selected_pts))
    
    # Create rectangles centered on points
    rects = [(x - w/2, y - h/2, x + w/2, y + h/2) 
            for (x, y), w, h in zip(selected_pts, widths, heights)]
    
    # Verify that each rectangle contains its center point
    for i, ((x, y), rect) in enumerate(zip(selected_pts, rects)):
        xmin, ymin, xmax, ymax = rect
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            print(f"Warning: Rectangle {i} doesn't contain its center point!")
    
    return rects

# Build and test functions
def build_rtree(pts, rtree_class, **kwargs):
    """Build an R-tree from points using the specified implementation."""
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

def visualize_query(tree, pts, rect, results, title):
    """Visualize a query with results."""
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
    ax.set_title(f"{title} ({len(results)} results)")
    
    return fig, ax

def evaluate_accuracy(tree_results, reference_results):
    """Calculate precision, recall, and F1 score against reference results."""
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

def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Generate data
    print("Generating data...")
    pts = sample_mixed(500)  # 1000 points (250 from each distribution)
    print(f"Generated {len(pts)} points")
    
    # Display some point samples
    print(f"Sample points: {pts[:3]}")
    
    # Generate queries
    small_queries = sample_data_based_rects(pts, 10, 10, 10)    # Small queries
    medium_queries = sample_data_based_rects(pts, 10, 50, 50)   # Medium queries
    large_queries = sample_data_based_rects(pts, 10, 200, 200)  # Large queries
    
    # Also add some random rectangles
    random_rects = sample_rects(10, 100, 100)
    
    all_queries = small_queries + medium_queries + large_queries + random_rects
    print(f"Generated {len(all_queries)} queries")
    
    # Print some sample queries
    for i, rect in enumerate(small_queries[:2]):
        print(f"Small query {i}: {rect}")
    
    # Manually verify that small query rectangles contain their center points
    for i, rect in enumerate(small_queries):
        # Find the original point this rectangle was based on
        # The center of the rectangle should be close to an original point
        xmin, ymin, xmax, ymax = rect
        center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
        
        # Find if any original point is within this rectangle
        contained_points = []
        for j, (x, y) in enumerate(pts):
            if xmin <= x <= xmax and ymin <= y <= ymax:
                contained_points.append(j)
        
        print(f"Query {i} contains {len(contained_points)} points. Center: ({center_x}, {center_y})")
        if contained_points:
            print(f"  First few contained points: {[pts[j] for j in contained_points[:3]]}")
    
    # Build trees
    print("\nBuilding trees...")
    rtree, rtree_time = build_rtree(pts, RTree, max_entries=CAPACITY)
    rstar, rstar_time = build_rtree(pts, RStarTree, max_entries=CAPACITY)
    ffm, ffm_time = build_rtree(pts, FisherFlowRTree, max_entries=CAPACITY)
    
    print(f"Build times (ms): Classic={rtree_time:.2f}, R*={rstar_time:.2f}, FFM={ffm_time:.2f}")
    
    # Debug: Manually test a query on the classic R-tree
    if small_queries:
        test_rect = small_queries[0]
        print(f"\nTesting manual query with rectangle: {test_rect}")
        rect_obj = Rect(*test_rect)
        
        # Manual query test
        results = list(rtree.query(rect_obj))
        print(f"Classic R-tree returns {len(results)} results")
        
        if results:
            result_ids = [e.data for e in results]
            result_pts = [pts[i] for i in result_ids if i < len(pts)]
            print(f"Result points: {result_pts}")
        else:
            # Debug the query intersection logic directly
            print("Debugging query intersection logic:")
            entries = list(rtree.get_leaf_entries())
            print(f"Tree has {len(entries)} leaf entries")
            
            # Check the first few entries
            for i, entry in enumerate(entries[:5]):
                print(f"Entry {i}: data={entry.data}, rect={entry.rect}")
                print(f"  Intersection test: {entry.rect.intersects(rect_obj)}")
    
    # Query trees
    print("\nQuerying trees...")
    rtree_results, rtree_query_time, rtree_hits, rtree_node_accesses = query_and_measure(rtree, all_queries)
    rstar_results, rstar_query_time, rstar_hits, rstar_node_accesses = query_and_measure(rstar, all_queries)
    ffm_results, ffm_query_time, ffm_hits, ffm_node_accesses = query_and_measure(ffm, all_queries)
    
    print(f"Query times (ms/query): Classic={rtree_query_time:.2f}, R*={rstar_query_time:.2f}, FFM={ffm_query_time:.2f}")
    
    # Calculate query hit statistics
    rtree_total_hits = sum(rtree_hits)
    rstar_total_hits = sum(rstar_hits)
    ffm_total_hits = sum(ffm_hits)
    
    print(f"Total hits: Classic={rtree_total_hits}, R*={rstar_total_hits}, FFM={ffm_total_hits}")
    
    # Report node access counts and pruning efficiency
    print(f"\nNode accesses: Classic={rtree_node_accesses}, R*={rstar_node_accesses}, FFM={ffm_node_accesses}")
    
    # Calculate pruning efficiency (% of nodes pruned compared to Guttman)
    total_nodes = len(list(rtree.get_nodes()))
    rstar_pruning = (1 - rstar_node_accesses / rtree_node_accesses) * 100
    ffm_pruning = (1 - ffm_node_accesses / rtree_node_accesses) * 100
    
    print(f"Pruning efficiency: R*={rstar_pruning:.2f}%, FFM={ffm_pruning:.2f}%")
    print(f"Total tree nodes: {total_nodes}")
    
    # Show hits per query type
    small_hits = sum(rtree_hits[:len(small_queries)])
    medium_hits = sum(rtree_hits[len(small_queries):len(small_queries)+len(medium_queries)])
    large_hits = sum(rtree_hits[len(small_queries)+len(medium_queries):len(small_queries)+len(medium_queries)+len(large_queries)])
    random_hits = sum(rtree_hits[len(small_queries)+len(medium_queries)+len(large_queries):])
    
    print(f"Hits by query type (Classic R-tree): Small={small_hits}, Medium={medium_hits}, Large={large_hits}, Random={random_hits}")
    
    # Evaluate accuracy (using Guttman as reference)
    print("\nEvaluating accuracy against Classic R-tree:")
    print("R*-Tree vs Classic:")
    rstar_prec, rstar_rec, rstar_f1 = evaluate_accuracy(rstar_results, rtree_results)
    
    print("\nFFM-Tree vs Classic:")
    ffm_prec, ffm_rec, ffm_f1 = evaluate_accuracy(ffm_results, rtree_results)
    
    # Test with specific queries that have results
    print("\nTesting specific queries with results...")
    results_found = False
    
    # Find a rectangle that has hits in the classic implementation
    for i, rect in enumerate(all_queries):
        if rtree_hits[i] > 0:
            print(f"Testing rectangle {i} with {rtree_hits[i]} classic hits...")
            rect_obj = Rect(*rect)
            
            # Query each implementation
            rtree_res = list(rtree.query(rect_obj))
            rstar_res = list(rstar.query(rect_obj))
            ffm_res = list(ffm.query(rect_obj))
            
            print(f"Results: Classic={len(rtree_res)}, R*={len(rstar_res)}, FFM={len(ffm_res)}")
            
            # Visualize results
            if len(rtree_res) > 0:
                results_found = True
                visualize_query(rtree, pts, rect, rtree_res, "Classic R-tree Query")
                plt.savefig("test_classic_query.png")
                
                visualize_query(rstar, pts, rect, rstar_res, "R*-tree Query")
                plt.savefig("test_rstar_query.png")
                
                visualize_query(ffm, pts, rect, ffm_res, "Fisher-Flow R-tree Query")
                plt.savefig("test_ffm_query.png")
                
                print("Saved query visualizations.")
                break
    
    if not results_found:
        print("No queries with results found for visualization.")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
benchmark_rtrees.py - Benchmark different R-tree implementations
"""

import time
import random
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.mixture import GaussianMixture
from scipy.stats import beta
from typing import List, Tuple, Dict, Any, Callable
import torch

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
        means = np.c_[np.random.uniform(XMIN, XMAX, k),
                      np.random.uniform(YMIN, YMAX, k)]
        gm = GaussianMixture(k, covariance_type="full", random_state=0)
        gm.fit(means)
        return gm.sample(n)[0]
    
    def skew(n, a=2, b=5):
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
    This ensures we get a mix of query sizes for more realistic testing.
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
    
    # Vary the width and height to create queries of different sizes
    widths = np.random.uniform(avg_width * 0.5, avg_width * 1.5, len(selected_pts))
    heights = np.random.uniform(avg_height * 0.5, avg_height * 1.5, len(selected_pts))
    
    return [(x - w/2, y - h/2, x + w/2, y + h/2) 
            for (x, y), w, h in zip(selected_pts, widths, heights)]

# Benchmarking functions
def build_rtree(pts, rtree_class, **kwargs):
    """Build an R-tree from points using the specified implementation."""
    tree = rtree_class(**kwargs)
    for i, (x, y) in enumerate(pts):
        tree.insert(i, Rect(x, y, x, y))
    return tree

def bench_query(tree, rects):
    """Benchmark query performance."""
    t0 = time.time()
    results = []
    
    # Reset node access counter
    tree.reset_node_accesses()
    
    print(f"Testing {len(rects)} queries...")
    total_results = 0
    
    for xmin, ymin, xmax, ymax in rects:
        query_rect = Rect(xmin, ymin, xmax, ymax)
        query_results = list(tree.query(query_rect))
        results.append(query_results)
        total_results += len(query_results)
    
    query_time = (time.time() - t0) / len(rects) * 1000
    print(f"Average results per query: {total_results/len(rects):.2f}")
    print(f"Total node accesses: {tree.node_accesses}")
    
    return query_time, results, tree.node_accesses

def bench_build(pts, rtree_class, **kwargs):
    """Benchmark tree building performance."""
    t0 = time.time()
    tree = build_rtree(pts, rtree_class, **kwargs)
    build_time = time.time() - t0
    return build_time * 1000, tree

def evaluate_accuracy(tree_results, reference_results):
    """Calculate precision, recall, and F1 score."""
    tp = fn = fp = 0
    empty_queries = 0
    matching_queries = 0
    total_queries = len(tree_results)
    
    for i, (tree_res, ref_res) in enumerate(zip(tree_results, reference_results)):
        tree_ids = set(e.data for e in tree_res)
        ref_ids = set(e.data for e in ref_res)
        
        # Count empty queries (no results in reference)
        if len(ref_ids) == 0:
            empty_queries += 1
            continue
            
        # Count exact matching queries
        if tree_ids == ref_ids:
            matching_queries += 1
        
        # Calculate TP, FP, FN
        query_tp = len(tree_ids & ref_ids)
        query_fp = len(tree_ids - ref_ids)
        query_fn = len(ref_ids - tree_ids)
        
        tp += query_tp
        fp += query_fp
        fn += query_fn
    
    # Calculate metrics
    non_empty_queries = total_queries - empty_queries
    prec = 1.0 if tp + fp == 0 else tp / (tp + fp)
    rec = 1.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    
    # Print detailed metrics
    print(f"Total queries: {total_queries}")
    print(f"Empty queries (no results in reference): {empty_queries}")
    print(f"Non-empty queries: {non_empty_queries}")
    print(f"Exactly matching queries: {matching_queries} ({100*matching_queries/non_empty_queries if non_empty_queries > 0 else 0:.2f}% of non-empty)")
    print(f"Accuracy metrics: TP={tp}, FP={fp}, FN={fn}")
    
    return prec, rec, f1

def visualize_rtree(tree, pts, rect=None, results=None, title=None):
    """Visualize an R-tree structure with optional query rectangle and results."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    
    # Plot all points
    xs, ys = zip(*pts)
    ax.scatter(xs, ys, s=4, c='lightgrey', alpha=0.5)
    
    # If query rectangle is provided, draw it
    if rect:
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
    
    # If results are provided, highlight them
    if results:
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
    else:
        ax.set_title('R-tree Visualization')
    
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
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    
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
        ax.set_xlim(XMIN, XMAX)
        ax.set_ylim(YMIN, YMAX)
    
    plt.tight_layout()
    return fig, axes

def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate data with different densities
    print("Generating data...")
    pts = sample_mixed(250)  # 1000 points (250 from each distribution)
    
    # Generate queries of different sizes
    # Use data-based rectangles to ensure we hit some points
    small_queries = sample_data_based_rects(pts, 20, 10, 10)    # Small queries
    medium_queries = sample_data_based_rects(pts, 20, 50, 50)   # Medium queries
    large_queries = sample_data_based_rects(pts, 20, 200, 200)  # Large queries
    
    # Also add some random rectangles to test areas without points
    random_small = sample_rects(10, 10, 10)
    random_medium = sample_rects(10, 50, 50)
    random_large = sample_rects(10, 200, 200)
    
    all_queries = small_queries + medium_queries + large_queries + random_small + random_medium + random_large
    print(f"Generated {len(pts)} points and {len(all_queries)} queries")
    
    # Build trees
    print("\nBuilding trees...")
    rtree_time, rtree = bench_build(pts, RTree, max_entries=CAPACITY)
    rstar_time, rstar = bench_build(pts, RStarTree, max_entries=CAPACITY)
    ffm_time, ffm = bench_build(pts, FisherFlowRTree, max_entries=CAPACITY)
    
    # Query performance
    print("\nBenchmarking queries...")
    rtree_query_time, rtree_results, rtree_accesses = bench_query(rtree, all_queries)
    rstar_query_time, rstar_results, rstar_accesses = bench_query(rstar, all_queries)
    ffm_query_time, ffm_results, ffm_accesses = bench_query(ffm, all_queries)
    
    # Calculate accuracy (Guttman RTree is the reference)
    print("\nEvaluating accuracy against Guttman RTree (reference):")
    print("\nR*-Tree vs Guttman:")
    rstar_prec, rstar_rec, rstar_f1 = evaluate_accuracy(rstar_results, rtree_results)
    
    print("\nFisher-Flow R-Tree vs Guttman:")
    ffm_prec, ffm_rec, ffm_f1 = evaluate_accuracy(ffm_results, rtree_results)
    
    # Print summary results
    print("\n===== BENCHMARK SUMMARY =====")
    print("\nBuild Time (ms):")
    print(f"{'RTree (Guttman)':20} {rtree_time:.2f}")
    print(f"{'RStarTree':20} {rstar_time:.2f}")
    print(f"{'FisherFlowRTree':20} {ffm_time:.2f}")
    
    print("\nQuery Time (ms/query):")
    print(f"{'RTree (Guttman)':20} {rtree_query_time:.2f}")
    print(f"{'RStarTree':20} {rstar_query_time:.2f}")
    print(f"{'FisherFlowRTree':20} {ffm_query_time:.2f}")
    
    # Add node access and pruning efficiency metrics
    print("\nNode Accesses:")
    print(f"{'RTree (Guttman)':20} {rtree_accesses}")
    print(f"{'RStarTree':20} {rstar_accesses}")
    print(f"{'FisherFlowRTree':20} {ffm_accesses}")
    
    # Calculate pruning efficiency compared to Guttman
    rstar_pruning = (1 - rstar_accesses / rtree_accesses) * 100
    ffm_pruning = (1 - ffm_accesses / rtree_accesses) * 100
    
    print("\nPruning Efficiency (% nodes pruned vs Guttman):")
    print(f"{'RTree (Guttman)':20} 0.00% (reference)")
    print(f"{'RStarTree':20} {rstar_pruning:.2f}%")
    print(f"{'FisherFlowRTree':20} {ffm_pruning:.2f}%")
    
    print("\nAccuracy (vs. Guttman RTree):")
    print(f"{'RTree (Guttman)':20} precision=100.00% recall=100.00% F1=100.00% (reference)")
    print(f"{'RStarTree':20} precision={rstar_prec:.2%} recall={rstar_rec:.2%} F1={rstar_f1:.2%}")
    print(f"{'FisherFlowRTree':20} precision={ffm_prec:.2%} recall={ffm_rec:.2%} F1={ffm_f1:.2%}")
    
    # Tree structure information
    print("\nTree Structure Information:")
    rtree_levels = len(rtree.get_levels())
    rstar_levels = len(rstar.get_levels())
    ffm_levels = len(ffm.get_levels())
    
    print(f"{'RTree (Guttman)':20} {rtree_levels} levels")
    print(f"{'RStarTree':20} {rstar_levels} levels")
    print(f"{'FisherFlowRTree':20} {ffm_levels} levels")
    
    # Visualize random query
    print("\nCreating visualizations...")
    # Choose a query that actually returns results
    random_rects = [rect for rect, results in zip(all_queries, rtree_results) if results]
    if random_rects:
        random_rect = random.choice(random_rects)
    else:
        random_rect = random.choice(all_queries)
    
    rtree_result = list(rtree.query(Rect(*random_rect)))
    rstar_result = list(rstar.query(Rect(*random_rect)))
    ffm_result = list(ffm.query(Rect(*random_rect)))
    
    print(f"Selected query rectangle: ({random_rect[0]:.1f}, {random_rect[1]:.1f}, {random_rect[2]:.1f}, {random_rect[3]:.1f})")
    print(f"Results: Guttman: {len(rtree_result)}, R*: {len(rstar_result)}, Fisher-Flow: {len(ffm_result)}")
    
    # Create visualizations
    # Compare tree structures
    fig, _ = compare_rtree_structures(
        [rtree, rstar, ffm],
        ["Guttman", "R*", "Fisher-Flow"],
        level=1
    )
    fig.savefig("rtree_structure_comparison_level1.png")
    print("Saved structure visualization: rtree_structure_comparison_level1.png")
    
    # Compare at level 2 (if available)
    max_level = min(rtree_levels, rstar_levels, ffm_levels) - 1
    if max_level >= 2:
        fig, _ = compare_rtree_structures(
            [rtree, rstar, ffm],
            ["Guttman", "R*", "Fisher-Flow"],
            level=2
        )
        fig.savefig("rtree_structure_comparison_level2.png")
        print("Saved structure visualization: rtree_structure_comparison_level2.png")
    
    # Visualize query results
    fig, ax = visualize_rtree(rtree, pts, random_rect, rtree_result, "Guttman RTree Query")
    fig.savefig("guttman_query.png")
    print("Saved query visualization: guttman_query.png")
    
    fig, ax = visualize_rtree(rstar, pts, random_rect, rstar_result, "R* Tree Query")
    fig.savefig("rstar_query.png")
    print("Saved query visualization: rstar_query.png")
    
    fig, ax = visualize_rtree(ffm, pts, random_rect, ffm_result, "Fisher-Flow RTree Query")
    fig.savefig("ffm_query.png")
    print("Saved query visualization: ffm_query.png")
    
    print("\nDone! Check the saved visualization files.")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Basic R-tree benchmark that directly compares the three rtreelib implementations.
"""

import sys
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

# Add parent directory to path to ensure we use the local package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtreelib import Rect, RTree, RStarTree, FisherFlowRTree

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Define our test data size
NUM_POINTS = 1000
NUM_QUERIES = 50
QUERY_SIZE = 10

# Create simple grid-like data for consistent testing
def create_grid_data(num_points, min_val=-100, max_val=100):
    """Create simple grid-like data for consistent testing"""
    side = int(np.sqrt(num_points))
    x = np.linspace(min_val, max_val, side)
    y = np.linspace(min_val, max_val, side)
    XX, YY = np.meshgrid(x, y)
    points = np.vstack([XX.ravel(), YY.ravel()]).T
    
    # Add some random jitter
    points += np.random.normal(0, 2, points.shape)
    
    # Ensure we have exactly num_points
    if len(points) > num_points:
        points = points[:num_points]
    elif len(points) < num_points:
        # Add some random points to fill up to num_points
        extra = num_points - len(points)
        extra_points = np.random.uniform(min_val, max_val, (extra, 2))
        points = np.vstack([points, extra_points])
    
    return points

def create_query_rects(points, num_queries, query_size):
    """Create query rectangles that are guaranteed to intersect with points"""
    queries = []
    
    # Create a mix of different query types
    
    # 1. Queries exactly around points
    for i in range(num_queries // 3):
        idx = random.randint(0, len(points) - 1)
        x, y = points[idx]
        half_size = query_size / 2
        queries.append((x - half_size, y - half_size, x + half_size, y + half_size))
    
    # 2. Queries that contain multiple points
    for i in range(num_queries // 3):
        idx = random.randint(0, len(points) - 1)
        x, y = points[idx]
        half_size = query_size * 2
        queries.append((x - half_size, y - half_size, x + half_size, y + half_size))
    
    # 3. Random queries
    for i in range(num_queries - len(queries)):
        x = random.uniform(-100, 100)
        y = random.uniform(-100, 100)
        half_size = query_size / 2
        queries.append((x - half_size, y - half_size, x + half_size, y + half_size))
    
    return queries

def benchmark_tree(tree_class, points, queries, name):
    """Benchmark a tree implementation"""
    # Build the tree
    start_time = time.time()
    tree = tree_class()
    for i, (x, y) in enumerate(points):
        tree.insert(i, Rect(x, y, x, y))
    build_time = (time.time() - start_time) * 1000  # ms
    
    # Query the tree
    start_time = time.time()
    results = []
    for xmin, ymin, xmax, ymax in queries:
        result = list(tree.query(Rect(xmin, ymin, xmax, ymax)))
        results.append(result)
    query_time = (time.time() - start_time) / len(queries) * 1000  # ms/query
    
    # Calculate statistics
    hits = sum(len(r) > 0 for r in results)
    total_results = sum(len(r) for r in results)
    
    print(f"\n--- {name} ---")
    print(f"Build time: {build_time:.2f} ms")
    print(f"Query time: {query_time:.2f} ms/query")
    print(f"Queries with hits: {hits}/{len(queries)} ({hits/len(queries)*100:.1f}%)")
    print(f"Total results: {total_results} (avg: {total_results/len(queries):.1f} per query)")
    
    return tree, results, build_time, query_time

def visualize_query_results(points, query_rect, results, tree_name):
    """Visualize the results of a query"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot all points
    ax.scatter(points[:, 0], points[:, 1], s=5, c='lightgray', alpha=0.5)
    
    # Plot the query rectangle
    xmin, ymin, xmax, ymax = query_rect
    rect = patches.Rectangle(
        (xmin, ymin), xmax - xmin, ymax - ymin, 
        linewidth=1, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)
    
    # Plot the results
    result_ids = [r.data for r in results]
    if result_ids:
        result_points = points[result_ids]
        ax.scatter(result_points[:, 0], result_points[:, 1], s=50, c='blue', marker='x')
    
    # Set title and limits
    ax.set_title(f"{tree_name} - {len(results)} results")
    ax.set_xlim(xmin - 20, xmax + 20)
    ax.set_ylim(ymin - 20, ymax + 20)
    
    return fig, ax

def main():
    print("Creating test data...")
    points = create_grid_data(NUM_POINTS)
    queries = create_query_rects(points, NUM_QUERIES, QUERY_SIZE)
    
    print(f"Generated {len(points)} points and {len(queries)} queries")
    
    # Benchmark each tree implementation
    guttman_tree, guttman_results, guttman_build_time, guttman_query_time = benchmark_tree(
        RTree, points, queries, "Guttman R-Tree"
    )
    
    rstar_tree, rstar_results, rstar_build_time, rstar_query_time = benchmark_tree(
        RStarTree, points, queries, "R*-Tree"
    )
    
    ffm_tree, ffm_results, ffm_build_time, ffm_query_time = benchmark_tree(
        FisherFlowRTree, points, queries, "Fisher-Flow R-Tree"
    )
    
    # Calculate accuracy compared to Guttman (reference implementation)
    print("\n--- Accuracy Comparison ---")
    
    def calculate_accuracy(test_results, reference_results):
        tp = fp = fn = 0
        for test_res, ref_res in zip(test_results, reference_results):
            test_ids = {r.data for r in test_res}
            ref_ids = {r.data for r in ref_res}
            
            tp += len(test_ids & ref_ids)
            fp += len(test_ids - ref_ids)
            fn += len(ref_ids - test_ids)
        
        precision = tp / (tp + fp) if tp + fp > 0 else 1.0
        recall = tp / (tp + fn) if tp + fn > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        
        return precision, recall, f1, tp, fp, fn
    
    rstar_precision, rstar_recall, rstar_f1, rstar_tp, rstar_fp, rstar_fn = calculate_accuracy(
        rstar_results, guttman_results
    )
    print(f"R*-Tree vs Guttman:")
    print(f"  TP={rstar_tp}, FP={rstar_fp}, FN={rstar_fn}")
    print(f"  Precision: {rstar_precision:.2%}, Recall: {rstar_recall:.2%}, F1: {rstar_f1:.2%}")
    
    ffm_precision, ffm_recall, ffm_f1, ffm_tp, ffm_fp, ffm_fn = calculate_accuracy(
        ffm_results, guttman_results
    )
    print(f"Fisher-Flow R-Tree vs Guttman:")
    print(f"  TP={ffm_tp}, FP={ffm_fp}, FN={ffm_fn}")
    print(f"  Precision: {ffm_precision:.2%}, Recall: {ffm_recall:.2%}, F1: {ffm_f1:.2%}")
    
    # Print summary
    print("\n--- Performance Summary ---")
    print(f"{'-'*50}")
    print(f"{'Guttman R-Tree':20} {guttman_build_time:15.2f} {guttman_query_time:15.2f}")
    print(f"{'R*-Tree':20} {rstar_build_time:15.2f} {rstar_query_time:15.2f}")
    print(f"{'Fisher-Flow R-Tree':20} {ffm_build_time:15.2f} {ffm_query_time:15.2f}")
    
    # Visualize a random query that has results
    for i in range(min(5, len(queries))):
        idx = random.randint(0, len(queries) - 1)
        if len(guttman_results[idx]) > 0:
            break
    
    print(f"\nVisualizing query {idx} with {len(guttman_results[idx])} results in Guttman tree")
    
    fig1, ax1 = visualize_query_results(points, queries[idx], guttman_results[idx], "Guttman R-Tree")
    fig1.savefig("guttman_query_basic.png")
    
    fig2, ax2 = visualize_query_results(points, queries[idx], rstar_results[idx], "R*-Tree")
    fig2.savefig("rstar_query_basic.png")
    
    fig3, ax3 = visualize_query_results(points, queries[idx], ffm_results[idx], "Fisher-Flow R-Tree")
    fig3.savefig("ffm_query_basic.png")
    
    print("Visualizations saved as guttman_query_basic.png, rstar_query_basic.png, and ffm_query_basic.png")

if __name__ == "__main__":
    main() 
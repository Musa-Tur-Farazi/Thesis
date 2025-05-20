# rtreelib

Pluggable R-tree implementation in pure Python.

## Overview

Since the original R-tree data structure has been initially proposed in 1984, there have been
many variations introduced over the years optimized for various use cases [1]. However, when
working in Python (one of the most popular languages for spatial data processing), there is
no easy way to quickly compare how these various implementations behave on real data.

The aim of this library is to provide a "pluggable" R-tree implementation that allows swapping
out the various strategies for insertion, node deletion, and other behaviors so that their
impact can be easily compared (without having to install separate libraries and having to
make code changes to accommodate for API differences). Several of the more common R-tree
variations will soon be provided as ready-built implementations (see the **Status** section below).

In addition, this library also provides utilities for inspecting the R-tree structure. It allows creating diagrams (using matplotlib and graphviz) that show the R-tree nodes and entries (including all the intermediate, non-leaf nodes), along with plots of their corresponding bounding boxes. It also allows exporting the R-tree to PostGIS so it could be examined using a GIS viewer like QGIS.

## Fisher-Flow R-tree (FFRtree)

This repository includes a new implementation that combines R*-tree with neural networks trained using flow matching to guide the traversal decisions. The Fisher-Flow R-tree aims to improve query performance by learning from the tree's structure during construction.

### How it works

1. The tree is built similar to an R*-tree initially
2. During insertion, we collect training data about which paths are chosen for each point
3. When enough training examples are collected, neural routers are trained for internal nodes
4. During query time, these routers help guide the search by considering the top-K most promising children
5. The result is potentially faster query performance with comparable accuracy

### Advantages

- Learns from the data distribution to make better traversal decisions
- Reduces the number of nodes visited during queries
- Combines classical R-tree algorithms with modern machine learning techniques
- Can adapt to different data distributions

### Usage

```python
from rtreelib import Rect, FisherFlowRTree

# Create a Fisher-Flow R-tree with default parameters
tree = FisherFlowRTree(max_entries=64)

# Insert points
for i, (x, y) in enumerate(points):
    tree.insert(i, Rect(x, y, x, y))

# Perform a range query
results = tree.query(Rect(x1, y1, x2, y2))
```

### Benchmarking

You can compare different R-tree implementations using the provided benchmarking script:

```
python scripts/benchmark_rtrees.py
```

This will:
1. Generate mixed distribution test data
2. Build trees using different R-tree variants (Guttman, R*, Fisher-Flow)
3. Compare query performance and accuracy
4. Generate visualizations of the tree structures and query results

## Status

This library is currently in early development. The table below shows which R-tree variants have been implemented, along with which operations they currently support:

| R-Tree Variant | Insert | Query | Delete |
|----------------|--------|-------|--------|
| Guttman        | ✓      | ✓     | ✓      |
| R*-Tree        | ✓      | ✓     | ✓      |
| Fisher-Flow    | ✓      | ✓     | ✗      |

## References

[1]: Nanopoulos, Alexandros & Papadopoulos, Apostolos (2003):
["R-Trees Have Grown Everywhere"](https://pdfs.semanticscholar.org/4e07/e800fe71505fbad686b08334abb49d41fcda.pdf)

[2]:  Guttman, A. (1984):
["R-trees: a Dynamic Index Structure for Spatial Searching"](http://www-db.deis.unibo.it/courses/SI-LS/papers/Gut84.pdf)
(PDF), *Proceedings of the 1984 ACM SIGMOD international conference on Management of data – SIGMOD
'84.* p. 47.

[3]: Beckmann, Norbert, et al.
["The R*-tree: an efficient and robust access method for points and rectangles."](https://infolab.usc.edu/csci599/Fall2001/paper/rstar-tree.pdf)
*Proceedings of the 1990 ACM SIGMOD international conference on Management of data.* 1990.

# Twitter Dataset Benchmark

In addition to the synthetic data experiments, we have added support for benchmarking with real-world Twitter data from the UCR STAR portal.

## Running the Twitter Dataset Benchmark

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Twitter dataset benchmark:
   ```
   cd scripts
   python benchmark_twitter.py
   ```

   The script will:
   - Automatically download the Twitter dataset from UCR STAR portal
   - Normalize the coordinates to a consistent space
   - Generate query rectangles of different sizes
   - Build and benchmark all three R-tree implementations
   - Generate visualizations of the results

3. If the automatic download fails, you can manually download the dataset:
   - Visit https://star.cs.ucr.edu/?Tweets
   - Click on "Download data" and select GeoJSON format
   - Save the file as `scripts/twitter_data.json`
   - Run the benchmark script again

## Twitter Benchmark Outputs

The benchmark will generate several output files:
- `twitter_rtree_structure_comparison_level1.png` - Comparison of R-tree structures
- `twitter_guttman_query.png` - Visualization of Guttman R-tree query results
- `twitter_rstar_query.png` - Visualization of R* tree query results
- `twitter_ffm_query.png` - Visualization of Fisher-Flow R-tree query results

The benchmark also prints detailed performance metrics comparing the three implementations.


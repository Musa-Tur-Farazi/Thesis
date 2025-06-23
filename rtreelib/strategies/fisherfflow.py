"""
Implementation of the Fisher-Flow R-Tree strategies, which uses flow-matching to guide R-tree traversal.
This approach combines classical R*-tree with neural networks trained using flow matching.
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, TypeVar, Tuple, Optional, Dict, Any, Set
from torch.utils.data import Dataset, DataLoader
from ..rtree import RTreeBase, RTreeEntry, RTreeNode, DEFAULT_MAX_ENTRIES, EPSILON
from rtreelib.models import Rect, Point, Location
from .base import adjust_tree_strategy, least_area_enlargement
from .rstar import rstar_split, rstar_choose_leaf, reinsert, _rstar_overflow

T = TypeVar('T')

# Default normalization bounds - these should be set based on your data
XMIN, XMAX, YMIN, YMAX = -1000.0, 1000.0, -1000.0, 1000.0
TOP_K_INNER = 2  # Number of children to consider during traversal

# Router network for flow matching
class RouterNet(nn.Module):
    """Neural network for routing decisions at internal nodes."""
    def __init__(self, k, hid=32, d=2):
        """
        Initialize the router network.
        
        Args:
            k: Number of output classes (usually the number of children)
            hid: Hidden layer dimensionality
            d: Depth of the network (number of hidden layers)
        """
        super().__init__()
        # Simpler architecture for faster inference
        self.net = nn.Sequential(
            nn.Linear(3, hid),  # Input: 2D point + time parameter
            nn.GELU(),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Linear(hid, k)
        )
        
    def forward(self, x, t):
        """Forward pass through the network."""
        return self.net(torch.cat([x, t], 1))
        
    def to_cpu(self):
        """Move model to CPU for inference"""
        self.to("cpu")
        return self

# Dataset for training the router
class RouterDS(Dataset):
    """Dataset for training router networks using flow matching."""
    def __init__(self, x_norm, child_ids, k):
        """
        Initialize the router dataset.
        
        Args:
            x_norm: Normalized coordinates of the points
            child_ids: Target child IDs for each point
            k: Number of possible children
        """
        self.x = torch.tensor(x_norm)
        self.cid = torch.tensor(child_ids)
        self.k = k
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        x = self.x[i]
        y = torch.zeros(self.k)
        y[self.cid[i]] = 1
        z0 = torch.rand(self.k)
        z0 /= z0.sum()
        eps = .05
        t = torch.rand(1) * (1 - 2 * eps) + eps
        h0, h1 = torch.sqrt(z0), torch.sqrt(y)
        cos = torch.clamp((h0 * h1).sum(), -1., 1.)
        th = torch.acos(cos) + 1e-9
        sin = lambda u: torch.sin(u)
        ht = (sin((1-t) * th) / sin(th)) * h0 + (sin(t * th) / sin(th)) * h1
        coef = (th / sin(th)) * (sin(t * th) * h1 - sin((1-t) * th) * h0)
        v = 2 * ht * coef
        return x.float(), t.float(), v.float()

def normalize(arr_like):
    """Normalize coordinates to [0,1] range based on global min/max."""
    a = np.asarray(arr_like, np.float32)
    return np.stack([
        (a[:, 0] - XMIN) / (XMAX - XMIN),
        (a[:, 1] - YMIN) / (YMAX - YMIN)
    ], 1)

def train_router(net, ds, tag, ep=5, lr=5e-3, device="cuda"):
    """
    Train a router network using flow matching.
    
    Args:
        net: RouterNet instance
        ds: RouterDS dataset
        tag: Name tag for logging
        ep: Number of epochs
        lr: Learning rate (increased for faster convergence)
        device: Device to train on ("cuda" or "cpu")
    """
    if not torch.cuda.is_available():
        device = "cpu"
    
    # Check if dataset is empty
    if len(ds) == 0:
        return
    
    # Always use CPU for faster training on small datasets
    device = "cpu"
        
    net.to(device)
    # Use higher learning rate for faster convergence
    opt = optim.Adam(net.parameters(), lr)
    
    # Better batch sizing strategy
    batch_size = min(len(ds), max(32, min(64, len(ds))))
    dl = DataLoader(ds, batch_size, shuffle=True, drop_last=False)
    
    # Early stopping setup
    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    # Train silently without printing every epoch
    for e in range(1, ep + 1):
        tot = cnt = 0
        for x, t, v in dl:
            if len(x) == 0:
                continue
                
            x, t, v = x.to(device), t.to(device), v.to(device)
            loss = ((net(x, t) - v) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            cnt += 1
        
        # Check for early stopping
        epoch_loss = tot/cnt if cnt > 0 else float('inf')
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Stop early if no improvement
        if patience_counter >= patience:
            break
    
    # Move back to CPU for inference
    net.to("cpu")
    net.eval()
    
    # Only output if final loss is high, indicating potential issues
    if best_loss > 0.1:
        print(f"[{tag}] training completed with final loss={best_loss:.4f}")

def ffm_choose_leaf(tree: RTreeBase[T], entry: RTreeEntry[T]) -> RTreeNode[T]:
    """
    Choose a leaf node using Fisher-Flow routing when available.
    Falls back to R* strategy when a router is not available.
    
    Args:
        tree: RTree instance
        entry: Entry to insert
        
    Returns:
        Leaf node for insertion
    """
    node = tree.root
    path = [node]
    
    while not node.is_leaf:
        if hasattr(node, 'router') and node.router and node.router[0] is not None and len(node.entries) > 0:
            try:
                # Use the neural router to make traversal decisions
                pt = entry.rect.centroid()
                
                # Fast torch-only evaluation (avoids detach/NumPy conversions)
                device = node.router[2]

                # Inline normalisation to keep allocations minimal
                coords = torch.tensor([
                    [(pt[0] - XMIN) / (XMAX - XMIN),
                     (pt[1] - YMIN) / (YMAX - YMIN)]
                ], device=device, dtype=torch.float32)

                # Use cached zero-tensor if available; otherwise create and store it
                if len(node.router) == 3:
                    # Append a cached tensor placeholder at index 3
                    node.router += (torch.zeros((1, 1), device=device, dtype=torch.float32),)
                t_zero = node.router[3]

                with torch.no_grad():
                    logits = node.router[0](coords, t_zero)
                best_idx = int(torch.argmax(logits, dim=1).item())
                
                # Ensure prediction is valid
                if best_idx < len(node.entries):
                    # Add training example if tree is Fisher-Flow implementation
                    if hasattr(tree, '_add_training_example'):
                        tree._add_training_example(node, pt)
                    
                    node = node.entries[best_idx].child
                else:
                    # Fall back to conventional method if prediction is invalid
                    e = least_area_enlargement(node.entries, entry.rect)
                    node = e.child
            except Exception:
                # If router fails, fall back to conventional method silently
                e = least_area_enlargement(node.entries, entry.rect)
                node = e.child
        else:
            # Fall back to R* strategy when router not available
            e = least_area_enlargement(node.entries, entry.rect)
            node = e.child
        
        path.append(node)
    
    return node

def _area_increase(mbr: Rect, entry_rect: Rect) -> float:
    """
    Calculate the area increase needed to include the entry in the MBR.
    
    Args:
        mbr: Existing minimum bounding rectangle
        entry_rect: Rectangle of the entry to insert
        
    Returns:
        Area increase after including entry
    """
    union_rect = mbr.union(entry_rect)
    return union_rect.area() - mbr.area()

def find_node_level(tree, node):
    """
    Find the level of a node in the tree safely.
    
    Args:
        tree: The R-tree
        node: The node to find the level for
        
    Returns:
        The level from leaf (0 = leaf level)
    """
    levels = tree.get_levels()
    for i, level in enumerate(levels):
        if node in level:
            return len(levels) - i - 1
    # If node not found, assume it's a leaf node (safest option)
    return 0

def ffm_overflow(tree: 'FisherFlowRTree', node: RTreeNode[T]) -> Optional[RTreeNode[T]]:
    """
    Custom overflow handler for Fisher-Flow R-tree.
    
    Args:
        tree: FisherFlowRTree instance
        node: Overflowing node
        
    Returns:
        Split node or None
    """
    # Initialize cache if not already done
    if not tree._cache:
        tree._cache = {}
    
    # Track nodes that have been processed for forced reinsert
    if 'reinsert' not in tree._cache:
        tree._cache['reinsert'] = {}
    
    # Find the level of this node from leaf
    level_from_leaf = find_node_level(tree, node)
    
    # If this is a root node or we've already done reinsert at this level,
    # just do a regular split
    if node.is_root or tree._cache['reinsert'].get(level_from_leaf, False):
        return rstar_split(tree, node)
    
    # Mark that we've done a reinsert at this level
    tree._cache['reinsert'][level_from_leaf] = True
    
    # Perform reinsert (similar to R*-tree)
    # Sort entries by distance from center
    center_x, center_y = node.get_bounding_rect().centroid()
    
    def distance(entry):
        entry_x, entry_y = entry.rect.centroid()
        dx, dy = entry_x - center_x, entry_y - center_y
        return dx*dx + dy*dy
    
    sorted_entries = sorted(node.entries, key=distance)
    
    # Remove 30% of entries for reinsertion
    p = math.ceil(0.3 * len(sorted_entries))
    entries_to_reinsert = sorted_entries[:p]
    
    # Remove entries from node
    node.entries = [e for e in node.entries if e not in entries_to_reinsert]
    
    # Update parent entry's rect
    if node.parent_entry:
        node.parent_entry.rect = union_all([entry.rect for entry in node.entries])
    
    # Reinsert entries
    for entry in entries_to_reinsert:
        leaf = tree.choose_leaf(tree, entry)
        leaf.entries.append(entry)
        if len(leaf.entries) > tree.max_entries:
            split_node = rstar_split(tree, leaf)
            tree.adjust_tree(tree, leaf, split_node)
        else:
            tree.adjust_tree(tree, leaf, None)
    
    return None

def ffm_insert(tree: 'FisherFlowRTree', data: T, rect: Rect) -> RTreeEntry[T]:
    """
    Insert an entry using Fisher-Flow strategy.
    
    Args:
        tree: FisherFlowRTree instance
        data: Entry data
        rect: Bounding rectangle
        
    Returns:
        The inserted entry
    """
    entry = RTreeEntry(rect, data=data)
    node = tree.choose_leaf(tree, entry)
    
    # Add training data for router (using rectangle centroid)
    if tree._is_training_enabled and not node.is_root:
        pt = rect.centroid()
        tree._add_training_example(node.parent, pt, node)
    
    node.entries.append(entry)
    split_node = None
    
    if len(node.entries) > tree.max_entries:
        split_node = tree.overflow_strategy(tree, node)
    
    tree.adjust_tree(tree, node, split_node)
    
    # Trigger router training if we have enough samples
    if tree._is_training_enabled and tree._training_buffer_size() > tree.min_train_samples:
        tree._train_routers()
    
    return entry

def union_all(rects):
    """
    Computes the union of all given rectangles.
    
    Args:
        rects: List of rectangles
        
    Returns:
        A rectangle that contains all input rectangles
    """
    if not rects:
        return None
    
    result = rects[0]
    for rect in rects[1:]:
        result = result.union(rect)
    return result

# noinspection PyProtectedMember
class FisherFlowRTree(RTreeBase[T]):
    """
    R-Tree implementation that uses Fisher-Flow strategies for traversal.
    This combines the R*-tree implementation with neural network routers.
    """
    
    def __init__(
        self, 
        max_entries: int = DEFAULT_MAX_ENTRIES, 
        min_entries: int = None,
        min_train_samples: int = 200,
        router_hidden_size: int = 64,  # Reduced from 64 for efficiency
        router_depth: int = 2,        # Reduced from 3 for efficiency
        router_epochs: int = 20,       # Reduced from 10 for efficiency
        router_top_k: int = 2,        # Changed from 2 to 1 for better pruning
        device: str = "cuda",
        training_enabled: bool = True
    ):
        """
        Initialize the Fisher-Flow R-Tree.
        
        Args:
            max_entries: Maximum entries per node
            min_entries: Minimum entries per node
            min_train_samples: Minimum samples needed to train a router
            router_hidden_size: Hidden layer size for router networks
            router_depth: Depth of router networks
            router_epochs: Training epochs for routers
            router_top_k: Number of children to consider during traversal
            device: Device for neural network training and inference
            training_enabled: Whether to enable router training
        """
        super().__init__(
            max_entries=max_entries,
            min_entries=min_entries,
            insert=ffm_insert,
            choose_leaf=ffm_choose_leaf,
            adjust_tree=adjust_tree_strategy,
            overflow_strategy=ffm_overflow
        )
        
        # Router hyperparameters
        self.min_train_samples = min_train_samples
        self.router_hidden_size = router_hidden_size
        self.router_depth = router_depth
        self.router_epochs = router_epochs
        self.router_top_k = router_top_k
        self.device = "cpu" if not torch.cuda.is_available() else device
        
        # Training buffer to collect samples for routers
        self._training_buffer = {}
        self._cache = {}
        self._is_training_enabled = training_enabled
        self._trained_nodes = set()  # Keep track of nodes we've already trained
    
    def _add_training_example(self, node: RTreeNode[T], point: Tuple[float, float], chosen_child=None) -> None:
        """
        Add a training example for a node's router.
        
        Args:
            node: The node where routing decision was made
            point: The point (usually centroid of the entry) being routed
            chosen_child: The chosen child node (if None, will find nearest)
        """
        if node not in self._training_buffer:
            self._training_buffer[node] = []
        
        # Find which child the point was assigned to
        if chosen_child is not None:
            # If we know which child was chosen
            nearest_child_idx = node.entries.index(next(e for e in node.entries if e.child is chosen_child))
        else:
            # If we need to compute it based on area increase
            nearest_child_idx = self._find_nearest_child(node, point)
            
        if nearest_child_idx is not None:
            self._training_buffer[node].append((normalize([point])[0], nearest_child_idx))
    
    def _find_nearest_child(self, node: RTreeNode[T], point: Tuple[float, float]) -> Optional[int]:
        """Find the index of the child that would be chosen for this point."""
        if not node.entries:
            return None
        
        point_rect = Rect(point[0], point[1], point[0], point[1])
        increases = [_area_increase(e.child.get_bounding_rect(), point_rect) for e in node.entries]
        return increases.index(min(increases))
    
    def _training_buffer_size(self) -> int:
        """Get the total number of training examples across all nodes."""
        return sum(len(examples) for examples in self._training_buffer.values())
    
    def _train_routers(self) -> None:
        """Train router networks for nodes with sufficient training data."""
        nodes_to_train = []
        samples_trained = 0
        
        # First, identify all nodes that need training
        for node, examples in list(self._training_buffer.items()):
            # Skip if already trained, not enough samples, or too few children
            if node in self._trained_nodes or len(examples) < self.min_train_samples or len(node.entries) <= 1:
                continue
            
            # Check for diverse training data (at least examples for 2 different children)
            child_ids = [ex[1] for ex in examples]
            unique_children = set(child_ids)
            if len(unique_children) < 2:
                continue
            
            # Ensure minimum examples per child (1% of min_train_samples)
            min_examples_per_child = max(1, int(0.01 * self.min_train_samples))
            counts = {child_id: child_ids.count(child_id) for child_id in unique_children}
            
            # For binary choices, we can be more lenient with training data requirements
            has_enough_examples = True
            if len(unique_children) > 2:
                for child_id, count in counts.items():
                    if count < min_examples_per_child:
                        has_enough_examples = False
                        break
            
            if has_enough_examples:
                nodes_to_train.append((node, examples))
        
        # If we have too many nodes to train, prioritize those with more examples
        if len(nodes_to_train) > 10:
            nodes_to_train.sort(key=lambda x: len(x[1]), reverse=True)
            nodes_to_train = nodes_to_train[:10]
            
        # Now train only the identified nodes
        for node, examples in nodes_to_train:
            try:
                # Prepare training data
                xs = np.array([ex[0] for ex in examples])
                child_ids = [ex[1] for ex in examples]
                
                # Create and train router
                net = RouterNet(len(node.entries), self.router_hidden_size, self.router_depth)
                dataset = RouterDS(xs, child_ids, len(node.entries))
                
                # Train the router without verbose output
                train_router(
                    net, 
                    dataset, 
                    "Router", 
                    ep=self.router_epochs, 
                    device=self.device
                )
                
                # Attach router to node
                node.router = (net, self.router_top_k, self.device)
                # Mark this node as trained to avoid unnecessary retraining
                self._trained_nodes.add(node)
                samples_trained += len(examples)
            except Exception:
                pass
        
        # Clear the buffer after training
        self._training_buffer.clear()
        
        # Only report if we trained significant data
        if samples_trained > 0:
            print(f"Trained {len(nodes_to_train)} routers with {samples_trained} total samples")
    
    def enable_training(self):
        """Enable router training"""
        self._is_training_enabled = True
        
    def disable_training(self):
        """Disable router training"""
        self._is_training_enabled = False 
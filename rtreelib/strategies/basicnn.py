import math
from typing import TypeVar, Tuple, List, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ..rtree import RTreeNode, RTreeEntry, DEFAULT_MAX_ENTRIES
from .fisherfflow import (
    FisherFlowRTree, RouterNet, normalize, ffm_insert,
    ffm_choose_leaf, adjust_tree_strategy, ffm_overflow
)

T = TypeVar('T')


class _BasicRouterDS(Dataset):
    """Simple classification dataset: given a point -> child id."""

    def __init__(self, xs: np.ndarray, child_ids: List[int]):
        self.x = torch.tensor(xs, dtype=torch.float32)
        self.cid = torch.tensor(child_ids, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Return x, dummy-t, target
        return self.x[idx], torch.zeros(1, dtype=torch.float32), self.cid[idx]


def _train_router_basic(net: RouterNet, ds: _BasicRouterDS, epochs: int = 10, lr: float = 1e-3,
                         device: str = "cpu") -> float:
    """Train router with cross-entropy on child id classification."""
    if not torch.cuda.is_available():
        device = "cpu"
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr)
    loader = DataLoader(ds, batch_size=min(64, len(ds)), shuffle=True, drop_last=False)
    net.train()
    best_loss = float('inf')
    patience = 3
    patience_counter = 0
    for epoch in range(1, epochs + 1):
        tot = cnt = 0
        for x, t, y in loader:
            x, t, y = x.to(device), t.to(device), y.to(device)
            logits = net(x, t)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            cnt += 1
        avg_loss = tot / max(1, cnt)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            break
    # Switch to eval / CPU
    net.to("cpu")
    net.eval()
    return best_loss


class BasicNNRTree(FisherFlowRTree):
    """R-Tree variant that trains a very simple classification router (no flow matching)."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES, min_entries: int = None,
                 min_train_samples: int = 200, router_hidden_size: int = 32,
                 router_depth: int = 1, router_epochs: int = 10, router_top_k: int = 1,
                 device: str = "cpu", training_enabled: bool = True):
        super().__init__(
            max_entries=max_entries,
            min_entries=min_entries,
            min_train_samples=min_train_samples,
            router_hidden_size=router_hidden_size,
            router_depth=router_depth,
            router_epochs=router_epochs,
            router_top_k=router_top_k,
            device=device,
            training_enabled=training_enabled
        )

    # Override to use simple cross-entropy instead of flow-matching training
    def _train_routers(self) -> None:  # type: ignore[override]
        nodes_to_train = []
        samples_trained = 0
        for node, examples in list(self._training_buffer.items()):
            if node in self._trained_nodes or len(examples) < self.min_train_samples or len(node.entries) <= 1:
                continue
            xs = np.array([ex[0] for ex in examples])
            child_ids = [ex[1] for ex in examples]
            # Need at least two different children to learn anything
            if len(set(child_ids)) < 2:
                continue
            nodes_to_train.append((node, xs, child_ids))
        for node, xs, child_ids in nodes_to_train:
            try:
                net = RouterNet(len(node.entries), self.router_hidden_size, self.router_depth)
                ds = _BasicRouterDS(xs, child_ids)
                loss = _train_router_basic(net, ds, epochs=self.router_epochs, device=self.device)
                node.router = (net, self.router_top_k, self.device)
                self._trained_nodes.add(node)
                samples_trained += len(child_ids)
            except Exception as e:
                print(f"[BasicNN] training error on node {id(node)}: {e}")
        # Clear buffer
        self._training_buffer.clear()
        if samples_trained:
            print(f"[BasicNN] trained {len(nodes_to_train)} routers with {samples_trained} samples (avg CE loss {loss:.4f})") 
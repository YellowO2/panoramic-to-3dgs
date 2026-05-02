import math
import numpy as np


def lat_lon_to_enu(lat: float, lon: float, origin_lat: float, origin_lon: float) -> np.ndarray:
    """Convert lat/lon to local ENU (East, North) in metres."""
    R = 6_371_000.0
    x = math.radians(lon - origin_lon) * R * math.cos(math.radians(origin_lat))
    z = math.radians(lat - origin_lat) * R
    return np.array([x, z])


def nearest_neighbor_order(nodes: list[dict]) -> list[str]:
    """Greedy nearest-neighbor chain over GPS positions. Returns ordered IDs."""
    if not nodes:
        return []
    n = len(nodes)
    origin = nodes[0]
    positions = np.array([
        lat_lon_to_enu(nd['lat'], nd['lon'], origin['lat'], origin['lon'])
        for nd in nodes
    ])
    visited = np.zeros(n, dtype=bool)
    order = [0]
    visited[0] = True
    for _ in range(n - 1):
        cur = order[-1]
        dists = np.linalg.norm(positions - positions[cur], axis=1)
        dists[visited] = np.inf
        nxt = int(np.argmin(dists))
        order.append(nxt)
        visited[nxt] = True
    ids = [nd['id'] for nd in nodes]
    return [ids[i] for i in order]


def make_batches(ordered_ids: list[str], batch_size: int = 4, overlap: int = 1) -> list[list[str]]:
    """Sliding-window batches with `overlap` shared panos between consecutive batches."""
    if len(ordered_ids) <= batch_size:
        return [list(ordered_ids)]
    stride = batch_size - overlap
    batches = []
    i = 0
    while i < len(ordered_ids):
        end = min(i + batch_size, len(ordered_ids))
        batches.append(ordered_ids[i:end])
        if end >= len(ordered_ids):
            break
        i += stride
    return batches

"""Reusable end-to-end test harness for run_pathfind_reconstruction: a
list of mock scenarios (nodes/edges/points/fail-pairs), each run through
the REAL pipeline code with DA3Model/_run_da3 mocked out, checked against
its own expected outcome.

Add a new scenario by appending a dict to SCENARIOS -- no new boilerplate
needed.
"""
import os
import sys
import types
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _stub_module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


_stub_module("components")
_stub_module("components.SplatGenerator")
_stub_module("components.SplatGenerator.SplatGenerator", SplatGenerator=object)
_stub_module("components.DepthMapGenerator")
_stub_module("components.DepthMapGenerator.DA3Model", DA3Model=object)
_stub_module("components.SplatProcessor")
_stub_module("components.SplatProcessor.SplatProcessor", SplatProcessor=object)
_stub_module("components.ViewExtractor")
_stub_module("components.ViewExtractor.ViewExtractor", extract_views=lambda *a, **k: None, extract_views_for_da3=lambda *a, **k: None)
_stub_module("components.Saver")
_stub_module("components.Saver.Saver", Saver=object)
_stub_module("components.SplatProcessor.utils", backproject_views_to_pcd=lambda *a, **k: None)
_stub_module("sharp")
_stub_module("sharp.utils")
_stub_module("sharp.utils.gaussians", Gaussians3D=object, save_ply=lambda *a, **k: None)

import panoramic_to_3dgs.pipeline as pipeline_mod

M_PER_DEG_LAT = 111320.0


def latlon(offset_m):
    """Fake positions along a single line, offset_m meters from a fixed
    origin -- real haversine math still applies, just along one axis."""
    return (1.0 + offset_m / M_PER_DEG_LAT, 103.0)


class FakeDA3Model:
    def __init__(self, *a, **kw):
        pass


def run_scenario(name, node_specs, edge_specs, point_offsets, fail_pairs, start_offset=0, **kwargs):
    """node_specs: {key: offset_m}. edge_specs: {key: [(other_key, dist_m), ...]}.
    point_offsets: [offset_m, ...] -- corridor spine. fail_pairs: set of
    frozenset({key_a, key_b}) that should always fail DA3's health check.
    Returns (segments, test_log)."""
    nodes = [(key, f"/fake/{key}", *latlon(off), date)
             for key, (off, date) in node_specs.items()]
    points = [latlon(off) for off in point_offsets]
    test_log = []

    def fake_run_da3(target_depth_path, support_paths, cfg, views_base, da3=None,
                      dist_thresh=0.2, angle_thresh=1, step_degrees=20):
        id_a = os.path.basename(target_depth_path)
        id_b = os.path.basename(support_paths[0])
        test_log.append((id_a, id_b))
        fails = frozenset({id_a, id_b}) in fail_pairs
        keep_counts = {id_a: (0 if fails else 1, 1), id_b: (0 if fails else 1, 1)}
        poses = {id_a: {"center": np.zeros(3), "rotation": np.eye(3)},
                 id_b: {"center": np.zeros(3), "rotation": np.eye(3)}}
        res = SimpleNamespace(pano_keep_counts=keep_counts, pano_poses=poses)
        return None, res, np.zeros((2, 3)), np.zeros((2, 3)), None, None

    pipeline_mod._run_da3 = fake_run_da3
    pipeline_mod.DA3Model = FakeDA3Model
    pipeline_mod.torch.cuda.empty_cache = lambda: None

    pipeline = pipeline_mod.Pipeline.__new__(pipeline_mod.Pipeline)
    pipeline.config = SimpleNamespace(da3_model="unused")
    start_lat, start_lon = latlon(start_offset)

    print(f"\n{'=' * 60}\nScenario: {name}\n{'=' * 60}")
    segments = pipeline.run_pathfind_reconstruction(
        nodes, edge_specs, points, start_lat, start_lon, **kwargs,
    )
    print(f"-> {len(test_log)} DA3 call(s): {test_log}")
    print(f"-> {len(segments)} segment(s): " +
          ", ".join(f"{d}({len(pe)} hops, {'full' if ra else 'partial'})" for _, _, pe, d, ra, _ in segments))
    return segments, test_log


# ---- scenarios ------------------------------------------------------
SCENARIOS = [
    {
        "name": "dead-end chain, one point unreachable by any date (live-log reproduction)",
        "node_specs": {
            "zF8": (0, "2022-05"), "4zMR": (10.8, "2022-05"), "A1jj": (21.6, "2022-05"),
        },
        "edge_specs": {
            "zF8": [("4zMR", 10.8)],
            "4zMR": [("zF8", 10.8), ("A1jj", 10.8)],
            "A1jj": [("4zMR", 10.8)],
        },
        "point_offsets": [0, 10.8, 21.6, 200],
        "fail_pairs": set(),
        "check": lambda segs, log: (
            len(log) == 2,
            f"expected exactly 2 DA3 calls, got {len(log)}: {log}",
        ),
    },
    {
        "name": "closest candidate fails, second choice works, then dead-ends; second date bridges the gap",
        "node_specs": {
            "n1": (0, "2022-05"), "n2_bad": (10, "2022-05"), "n2": (10, "2022-05"), "n3": (20, "2022-05"),
            "m1": (0, "2020-07"), "m2": (50, "2020-07"),
        },
        "edge_specs": {
            "n1": [("n2_bad", 10.0), ("n2", 10.0)],
            "n2_bad": [("n1", 10.0)],
            "n2": [("n1", 10.0), ("n3", 10.0)],
            "n3": [("n2", 10.0)],
            "m1": [("m2", 50.0)],
            "m2": [("m1", 50.0)],
        },
        "point_offsets": [0, 10, 20, 50],
        "fail_pairs": {frozenset({"n1", "n2_bad"})},
        "check": lambda segs, log: (
            sum(1 for e in log if frozenset(e) == frozenset({"n1", "n2_bad"})) == 1
            and len(log) == 4
            and {s[3] for s in segs} == {"2022-05", "2020-07"}
            and any(s[4] for s in segs),
            f"expected 4 calls total (dead edge once), both dates combined, full coverage; got {len(log)} calls: {log}, dates: {[s[3] for s in segs]}",
        ),
    },
    {
        # Spacing matters: node offsets must clear point_cover_tolerance_m
        # (default 15m) and start_zone_m (default 5m) from each other, or
        # tolerance-based over-coverage or extra roots silently change what
        # the scenario is actually testing (hit this the first pass: a 7m-
        # wide version let one node cover everything by proximity alone).
        "name": "two separate gaps on one date (x-x-o-o-o-x), bridged by a different pairing",
        "node_specs": {
            "N0": (0, "A"), "N1": (10, "A"), "N2": (60, "A"), "N3": (100, "A"), "N4": (140, "A"),
        },
        "edge_specs": {
            "N0": [("N1", 10.0), ("N2", 60.0)],
            "N1": [("N0", 10.0), ("N3", 90.0)],
            "N2": [("N0", 60.0), ("N3", 40.0)],
            "N3": [("N2", 40.0), ("N1", 90.0), ("N4", 40.0)],
            "N4": [("N3", 40.0)],
        },
        "point_offsets": [0, 10, 60, 100, 140],
        "fail_pairs": {frozenset({"N0", "N1"})},
        "check": lambda segs, log: (
            sum(1 for e in log if frozenset(e) == frozenset({"N0", "N1"})) == 1
            and len(log) <= 5
            and any(s[4] for s in segs),
            f"expected dead edge tested once, <=5 calls total, full coverage; got {len(log)} calls: {log}",
        ),
    },
]


def run_fuzz(seed, fail_rate, n_dates=3, nodes_per_date=6, spacing_m=15.0):
    """Randomized stress test: a synthetic multi-date graph (chain topology
    per date, some extra cross-links), DA3 pass/fail decided by a coin flip
    PER PAIR (same pair always gets the same outcome, since real DA3 would
    give the same answer if re-tested -- this is what would catch a
    duplicate-testing regression). Returns (segments, test_log, tested_pairs).
    Raises whatever the real pipeline raises -- fuzzing is exactly for
    catching crashes a curated scenario wouldn't think to construct."""
    import random
    rng = random.Random(seed)

    node_specs = {}
    edge_specs = {}
    for d in range(n_dates):
        date = f"date{d}"
        offsets = sorted(rng.uniform(0, (nodes_per_date - 1) * spacing_m) for _ in range(nodes_per_date))
        keys = [f"d{d}n{i}" for i in range(nodes_per_date)]
        for k, off in zip(keys, offsets):
            node_specs[k] = (off, date)
            edge_specs[k] = []
        # chain (guarantees connectivity) + a few random extra same-date edges
        for i in range(nodes_per_date - 1):
            dist = offsets[i + 1] - offsets[i]
            edge_specs[keys[i]].append((keys[i + 1], dist))
            edge_specs[keys[i + 1]].append((keys[i], dist))
        for _ in range(nodes_per_date // 2):
            i, j = rng.sample(range(nodes_per_date), 2)
            dist = abs(offsets[i] - offsets[j])
            edge_specs[keys[i]].append((keys[j], dist))
            edge_specs[keys[j]].append((keys[i], dist))

    point_offsets = sorted(off for off, _ in node_specs.values())

    tested_pairs = {}

    def fake_run_da3(target_depth_path, support_paths, cfg, views_base, da3=None,
                      dist_thresh=0.2, angle_thresh=1, step_degrees=20):
        id_a = os.path.basename(target_depth_path)
        id_b = os.path.basename(support_paths[0])
        pair = frozenset({id_a, id_b})
        if pair not in tested_pairs:
            tested_pairs[pair] = rng.random() >= fail_rate
        ok = tested_pairs[pair]
        keep_counts = {id_a: (1 if ok else 0, 1), id_b: (1 if ok else 0, 1)}
        poses = {id_a: {"center": np.zeros(3), "rotation": np.eye(3)},
                 id_b: {"center": np.zeros(3), "rotation": np.eye(3)}}
        res = SimpleNamespace(pano_keep_counts=keep_counts, pano_poses=poses)
        return None, res, np.zeros((2, 3)), np.zeros((2, 3)), None, None

    test_log = []
    real_fake = fake_run_da3

    def logging_fake(*a, **k):
        target_depth_path, support_paths = a[0], a[1]
        test_log.append((os.path.basename(target_depth_path), os.path.basename(support_paths[0])))
        return real_fake(*a, **k)

    pipeline_mod._run_da3 = logging_fake
    pipeline_mod.DA3Model = FakeDA3Model
    pipeline_mod.torch.cuda.empty_cache = lambda: None

    nodes = [(key, f"/fake/{key}", *latlon(off), date) for key, (off, date) in node_specs.items()]
    points = [latlon(off) for off in point_offsets]
    pipeline = pipeline_mod.Pipeline.__new__(pipeline_mod.Pipeline)
    pipeline.config = SimpleNamespace(da3_model="unused")
    start_lat, start_lon = latlon(point_offsets[0])

    segments = pipeline.run_pathfind_reconstruction(
        nodes, edge_specs, points, start_lat, start_lon, top_n_dates=n_dates,
    )
    return segments, test_log, tested_pairs


if __name__ == "__main__":
    import contextlib
    import io

    failures = []
    for sc in SCENARIOS:
        segs, log = run_scenario(
            sc["name"], sc["node_specs"], sc["edge_specs"], sc["point_offsets"], sc["fail_pairs"],
            top_n_dates=5, early_exit_segments=4,
        )
        ok, msg = sc["check"](segs, log)
        status = "PASS" if ok else "FAIL"
        print(f"-> {status}: {msg}" if not ok else f"-> {status}")
        if not ok:
            failures.append(sc["name"])

    print(f"\n{'=' * 60}\nFuzz: randomized graphs x randomized DA3 pass/fail\n{'=' * 60}")
    fuzz_failures = []
    for fail_rate in (0.2, 0.4, 0.6, 0.8):
        crashes = 0
        dupes = 0
        max_calls = 0
        for seed in range(30):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    segs, log, tested_pairs = run_fuzz(seed=seed * 1000 + int(fail_rate * 10), fail_rate=fail_rate)
            except Exception as e:
                crashes += 1
                print(f"  fail_rate={fail_rate} seed={seed}: CRASH {type(e).__name__}: {e}")
                continue
            dup = len(log) - len(set(frozenset(e) for e in log))
            dupes += dup
            max_calls = max(max_calls, len(log))
        print(f"fail_rate={fail_rate}: 30 seeds, crashes={crashes}, duplicate_calls={dupes}, max_calls_seen={max_calls}")
        if crashes or dupes:
            fuzz_failures.append(fail_rate)

    print(f"\n{'=' * 60}")
    if failures or fuzz_failures:
        if failures:
            print(f"{len(failures)}/{len(SCENARIOS)} curated scenario(s) FAILED: {failures}")
        if fuzz_failures:
            print(f"fuzz FAILED at fail_rate(s): {fuzz_failures}")
        sys.exit(1)
    print(f"All {len(SCENARIOS)} curated scenarios + fuzz (120 randomized runs) passed.")

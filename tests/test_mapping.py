from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cave_mapper import map_points, mesh_point_offset, points_to_voxels

ROOT = 864691135339247078
OTHER_ROOT = 864691135339247079
# 64-bit ids, past float64 precision, so a float cast anywhere would corrupt them.
ROOT_SV = 106342634603503503
OTHER_SV = 106342634603503504
RES = (8, 8, 40)
RESOLUTIONS = {0: (8, 8, 40), 1: (16, 16, 40), 2: (32, 32, 40)}


class FakeCloudVolume:
    """Voxels with x >= `root_from_x` belong to ROOT, others to OTHER_ROOT."""

    def __init__(self, root_from_x=10, mesh_info=None):
        self.root_from_x = root_from_x
        self.queried = []
        self.meta = SimpleNamespace(resolution=lambda mip: np.array(RESOLUTIONS[mip]))
        self.mesh = SimpleNamespace(
            meta=SimpleNamespace(info={"mip": 2} if mesh_info is None else mesh_info)
        )

    def scattered_points(self, pts, coord_resolution=None, agglomerate=None):
        factor = np.array(RES) / np.array(coord_resolution)
        out = {}
        for pt in pts:
            voxel = tuple(int(v) for v in np.floor(np.asarray(pt) / factor))
            self.queried.append(voxel)
            out[voxel] = ROOT_SV if voxel[0] >= self.root_from_x else OTHER_SV
        return out


def fake_client(cv):
    calls = {"timestamps": 0, "get_roots": []}

    def get_root_timestamps(root_id, latest=False):
        calls["timestamps"] += 1
        return [pd.Timestamp("2026-01-01", tz="UTC")]

    def get_roots(svids, timestamp=None):
        calls["get_roots"].append(timestamp)
        lookup = {ROOT_SV: ROOT, OTHER_SV: OTHER_ROOT}
        return np.array([lookup[int(s)] for s in svids], dtype=np.int64)

    client = SimpleNamespace(
        info=SimpleNamespace(segmentation_cloudvolume=lambda **kwargs: cv),
        chunkedgraph=SimpleNamespace(
            get_root_timestamps=get_root_timestamps, get_roots=get_roots
        ),
    )
    return client, calls


def voxel_point(voxel, within=(0.5, 0.5, 0.5)):
    """A point in nm inside `voxel`, at fraction `within` of the way across it."""
    return (np.array(voxel) + np.array(within)) * np.array(RES)


def test_first_try_hit():
    cv = FakeCloudVolume()
    client, _ = fake_client(cv)
    out = map_points([voxel_point((12, 5, 3))], ROOT, client, convention="floor")
    row = out.iloc[0]
    assert (row["voxel_pt_x"], row["voxel_pt_y"], row["voxel_pt_z"]) == (12, 5, 3)
    assert row["search_distance"] == 0
    assert row["supervoxel_id"] == ROOT_SV
    assert row["root_id"] == ROOT


def test_anchor_floors_back_to_voxel():
    cv = FakeCloudVolume()
    client, _ = fake_client(cv)
    points = [voxel_point((12, 5, 3)), voxel_point((9, 5, 3))]
    out = map_points(points, ROOT, client, convention="floor")
    anchor = out[["anchor_pt_x", "anchor_pt_y", "anchor_pt_z"]].to_numpy()
    voxel = out[["voxel_pt_x", "voxel_pt_y", "voxel_pt_z"]].to_numpy(dtype=np.int64)
    np.testing.assert_array_equal(points_to_voxels(anchor, RES, "floor"), voxel)
    # The materialization engine's own lookup: an annotation at (4, 4, 40) nm.
    in_table_units = anchor / np.array((4, 4, 40))
    np.testing.assert_array_equal(
        np.floor(in_table_units * np.array((4, 4, 40)) / np.array(RES)), voxel
    )


def test_miss_found_on_first_shell():
    cv = FakeCloudVolume(root_from_x=10)
    client, _ = fake_client(cv)
    out = map_points([voxel_point((9, 5, 3))], ROOT, client, convention="floor")
    row = out.iloc[0]
    assert row["search_distance"] == 1
    assert row["voxel_pt_x"] == 10
    assert row["supervoxel_id"] == ROOT_SV
    # The nearest voxel of the root is straight across in x, one voxel away.
    assert row["voxel_pt_y"] == 5 and row["voxel_pt_z"] == 3
    assert row["query_voxel_distance_nm"] == pytest.approx(8.0)


def test_not_found_is_null():
    cv = FakeCloudVolume(root_from_x=100)
    client, _ = fake_client(cv)
    out = map_points(
        [voxel_point((9, 5, 3)), voxel_point((150, 5, 3))],
        ROOT,
        client,
        max_distance=2,
        convention="floor",
    )
    assert out.loc[0, ["supervoxel_id", "root_id", "voxel_pt_x"]].isna().all()
    assert np.isnan(out.loc[0, "anchor_pt_x"])
    assert out.loc[1, "supervoxel_id"] == ROOT_SV
    assert out["supervoxel_id"].dtype == "Int64"


def test_floor_against_round():
    # 9.75 voxels in x: floor gives 9, round gives 10.
    point = [[9.75 * 8, 5.5 * 8, 3.5 * 40]]
    for convention, expected in [("floor", 9), ("round", 10)]:
        cv = FakeCloudVolume(root_from_x=0)
        client, _ = fake_client(cv)
        out = map_points(point, ROOT, client, convention=convention)
        assert out.loc[0, "voxel_pt_x"] == expected


def test_shells_never_repeat_a_voxel():
    cv = FakeCloudVolume(root_from_x=100)
    client, _ = fake_client(cv)
    map_points([voxel_point((9, 5, 3))], ROOT, client, max_distance=2)
    assert len(cv.queried) == len(set(cv.queried)) == 5**3


def test_timestamp_fetched_once():
    cv = FakeCloudVolume(root_from_x=100)
    client, calls = fake_client(cv)
    map_points([voxel_point((9, 5, 3))], ROOT, client, max_distance=3)
    assert calls["timestamps"] == 1
    assert len(calls["get_roots"]) == 4
    assert len(set(calls["get_roots"])) == 1


def test_given_timestamp_is_used():
    cv = FakeCloudVolume()
    client, calls = fake_client(cv)
    when = pd.Timestamp("2025-06-01", tz="UTC")
    map_points([voxel_point((12, 5, 3))], ROOT, client, timestamp=when)
    assert calls["timestamps"] == 0
    assert calls["get_roots"] == [when]


def test_mesh_offset():
    cv = FakeCloudVolume(root_from_x=0)
    np.testing.assert_array_equal(mesh_point_offset(cv), [16, 16, 20])
    client, _ = fake_client(cv)
    # 0.75 of the way across voxel 9 in x; +16 nm (two voxels) moves it into voxel 11.
    out = map_points(
        [voxel_point((9, 5, 3), within=(0.75, 0.5, 0.25))],
        ROOT,
        client,
        point_offset="mesh",
        convention="floor",
    )
    assert out.loc[0, "voxel_pt_x"] == 11
    assert out.loc[0, "voxel_pt_z"] == 3


def test_mesh_offset_needs_mesh_mip():
    with pytest.raises(ValueError, match="mip"):
        mesh_point_offset(FakeCloudVolume(mesh_info={}))

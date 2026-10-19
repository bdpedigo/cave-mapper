from typing import Optional

import numpy as np
import pandas as pd
import point_cloud_utils as pcu
from caveclient import CAVEclient
from cloudvolume import CloudVolume


def _get_vertex_order(mesh: tuple) -> str:
    """
    Determine the order of the vertices in the mesh.
    """
    v = mesh[0]  # vertices of the mesh
    if v.flags["C_CONTIGUOUS"]:
        v_order = "C"
    else:
        v_order = "F"
    return v_order


def find_closest_points_on_mesh(points: np.ndarray, mesh: tuple) -> np.ndarray:
    vertices = mesh[0]
    v_order = _get_vertex_order(mesh)
    faces = mesh[1]

    points = np.array(points, dtype=vertices.dtype, order=v_order)

    _, fid, bc = pcu.closest_points_on_mesh(points, vertices, faces)

    use_barycentric = True
    if use_barycentric:
        # Interpolate the barycentric coordinates to get the coordinates of
        # the closest points on the mesh to each point in p
        closest_pts = pcu.interpolate_barycentric_coords(faces, fid, bc, vertices)
    else:
        closest_face_index = np.argmax(bc, axis=1)
        node_index = faces[fid, closest_face_index]
        closest_pts = vertices[node_index]

    return closest_pts


def _root_timestamp(client: CAVEclient, root_id: int):
    """A time at which `root_id` is valid: the end of its lifetime, or now if it is
    still the latest root. Any time in that window gives the same `get_roots`."""
    return client.chunkedgraph.get_root_timestamps(root_id, latest=True)[0]


def mesh_point_offset(cv: CloudVolume) -> np.ndarray:
    """
    The offset, in nanometers, to add to a point taken from a mesh before looking
    up the voxel that holds it.

    Meshes are built from a downsampled segmentation (the mesh "mip") and sit half
    a voxel of that mip off the voxel grid. This returns that half voxel,
    `resolution(mesh_mip) / 2`.

    Measured on MICrONS (`minnie65_phase3_v1`, mesh mip 2): the shift is about
    (15, 8, 19) nm, against (16, 16, 20) from this rule. With the rule and
    `convention="floor"`, deep points inside spine meshes landed on the root
    100% and 99.8% of the time on three cells, against 96% to 98% for medoids
    with `round`. On H01 (`h01_c3_flat`, mesh mip 2) the measured shift is about
    (3, 6, 15) nm, against (16, 16, 16.5). The rule still gave 99.58% there,
    against 99.68% for the measured offset and 99.34% for none.

    Parameters
    ----------
    cv :
        The segmentation CloudVolume. Its mesh info must name the mesh `mip`.

    Returns
    -------
    :
        Array of shape (3,), in nanometers.
    """
    info = cv.mesh.meta.info
    if "mip" not in info:
        raise ValueError(
            "The mesh info of this CloudVolume names no 'mip', so the mesh offset "
            "is unknown; pass point_offset explicitly."
        )
    return np.asarray(cv.meta.resolution(info["mip"]), dtype=float) / 2


def points_to_voxels(
    points: np.ndarray, resolution: np.ndarray, convention: str = "round"
) -> np.ndarray:
    """
    Convert points in nanometers to mip-0 voxel indices.

    `convention="floor"` is the voxel that holds the point: voxel `i` covers
    `[i * res, (i + 1) * res)`. CloudVolume, Neuroglancer and the CAVE
    materialization engine all use it. `convention="round"` is the older behavior
    of this package, which treats `i * res` as the voxel center.
    """
    if convention not in ("floor", "round"):
        raise ValueError(f"convention must be 'floor' or 'round', not {convention!r}")
    points = np.asarray(points, dtype=float)
    if np.isnan(points).any():
        raise ValueError("Points contain NaN values, which cannot be used for mapping.")
    scaled = points / np.asarray(resolution, dtype=float)
    rounded = np.floor(scaled) if convention == "floor" else np.round(scaled)
    return rounded.astype(np.int64)


def shell_offsets(distance: int) -> np.ndarray:
    """The voxel offsets at exactly Chebyshev distance `distance` from the origin."""
    cube = construct_neighbor_elements(distance, distance, distance)
    return cube[np.abs(cube).max(axis=1) == distance]


def cast_points_to_chunked(points: np.ndarray, cv: CloudVolume) -> np.ndarray:
    """
    Cast points to the chunked representation of the mesh.
    """
    points = np.array(points, dtype=np.float32)
    if np.isnan(points).any():
        raise ValueError("Points contain NaN values, which cannot be used for mapping.")
    points_cast = points / np.array(cv.meta.resolution(0))
    points_cast = np.round(points_cast).astype(int)
    return points_cast


def scattered_points(
    points: np.ndarray,
    cv: CloudVolume,
    client: CAVEclient,
    root_id: Optional[int] = None,
    timestamp=None,
) -> pd.DataFrame:
    out = cv.scattered_points(
        points, coord_resolution=cv.meta.resolution(0), agglomerate=False
    )

    scatter_df = (
        pd.DataFrame.from_dict(out, orient="index", columns=["supervoxel_id"])
        .rename_axis("point")
        .reset_index()
    )
    scatter_df["x"] = scatter_df["point"].apply(lambda x: x[0])
    scatter_df["y"] = scatter_df["point"].apply(lambda x: x[1])
    scatter_df["z"] = scatter_df["point"].apply(lambda x: x[2])

    scatter_df = (
        scatter_df.set_index(["x", "y", "z"])
        .loc[pd.MultiIndex.from_arrays(points.T)]
        .drop(columns=["point"])
    )

    if timestamp is None and root_id is not None:
        timestamp = _root_timestamp(client, root_id)

    root_lookups = client.chunkedgraph.get_roots(
        scatter_df["supervoxel_id"].values, timestamp=timestamp
    )

    scatter_df["root_id"] = root_lookups

    return scatter_df


def construct_neighbor_elements(max_x, max_y, max_z) -> np.ndarray:
    xs, ys, zs = np.nonzero(
        np.ones((1 + 2 * max_x, 1 + 2 * max_y, 1 + 2 * max_z), dtype=bool)
    )
    xs = xs - max_x
    ys = ys - max_y
    zs = zs - max_z
    neighbors = np.array([xs, ys, zs]).T
    return neighbors


def construct_lookup_points(
    cast_closest_points: np.ndarray, neighbors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lookup_matrix = np.repeat(
        cast_closest_points.reshape((*cast_closest_points.shape, 1)),
        len(neighbors),
        axis=2,
    )
    lookup_matrix = lookup_matrix + neighbors.T[None, :, :]

    index_matrix = np.repeat(
        np.arange(cast_closest_points.shape[0]).reshape(
            (cast_closest_points.shape[0], 1)
        ),
        neighbors.shape[0],
        axis=1,
    )

    # lookup matrix is n_points x 3 x n_neighbors
    # want it to be n_points * n_neighbors x 3
    lookup_matrix = lookup_matrix.swapaxes(1, 2)
    # now tile it to be n_points * n_neighbors x 3
    lookup_points = lookup_matrix.reshape(
        (lookup_matrix.shape[0] * lookup_matrix.shape[1], lookup_matrix.shape[2])
    )

    index_points = index_matrix.reshape((index_matrix.shape[0] * index_matrix.shape[1]))

    return lookup_points, index_points


def map_points_via_mesh(
    points: np.ndarray,
    root_id: int,
    client: CAVEclient,
    max_distance: int = 4,
    verbose=False,
) -> pd.DataFrame:
    """
    Map points in space to a mesh and then to voxels and their corresponding supervoxels.

    Parameters
    ----------
    points :
        Array of points in space to be mapped. Must be of shape (n_points, 3), and
        are assumed to be in coordinates of nanometers.
    root_id :
        The root id of the object to map to.
    client :
        The CAVEclient object to use for mapping.
    max_distance :
        The maximum distance to search for a mapping. Default is 4. Units are in voxels.
    verbose :
        Whether to print progress messages.

    Returns
    -------
    :
        A DataFrame containing the mapping information. The DataFrame will have the
        following columns:

        - `query_pt_x`, `query_pt_y`, `query_pt_z`: The original points in space.
        - `mesh_pt_x`, `mesh_pt_y`, `mesh_pt_z`: The closest points on the mesh.
        - `voxel_pt_x`, `voxel_pt_y`, `voxel_pt_z`: The corresponding voxel coordinates.
            Units are the same as `client.chunkedgraph.base_resolution`.
        - `supervoxel_id`: The supervoxel id of the corresponding voxel.
        - `root_id`: The root id of the object.
        - `query_mesh_distance_nm`: The distance from the original point to the closest
            point on the mesh.
        - `mesh_voxel_distance_nm`: The distance from the closest point on the mesh to
            the corresponding voxel.

    """
    cv: CloudVolume = client.info.segmentation_cloudvolume(progress=verbose)

    res = np.array(cv.meta.resolution(0))

    raw_mesh = cv.mesh.get(
        root_id, remove_duplicate_vertices=True, deduplicate_chunk_boundaries=False
    )[root_id]

    mesh = (raw_mesh.vertices, raw_mesh.faces)
    vertices: np.ndarray = mesh[0]
    v_order = _get_vertex_order(mesh)
    faces: np.ndarray = mesh[1]

    closest_pts = find_closest_points_on_mesh(points, mesh)

    cast_closest_points = cast_points_to_chunked(closest_pts, cv)
    timestamp = _root_timestamp(client, root_id)

    distance = 1

    point_mapping = {}
    info = []
    missing_point_ids = np.arange(len(cast_closest_points))
    while len(point_mapping) < len(closest_pts) and distance <= max_distance:
        missing_points = cast_closest_points[missing_point_ids]

        # find the k-vertex neighbors of the missing points
        neighbors = construct_neighbor_elements(distance, distance, distance)
        lookup_points, index_points = construct_lookup_points(missing_points, neighbors)

        # remove points that are not even in the mesh
        # this takes a few seconds of overhead per check, thus sometimes not worth it
        # does seem pretty flat in terms of number of verts for the check, though, so
        # maybe could batch these ahead of time
        if len(lookup_points) > 1000:
            points_for_sdf = lookup_points * np.array(cv.meta.resolution(0)).astype(
                vertices.dtype
            )
            points_for_sdf = np.array(
                points_for_sdf, dtype=vertices.dtype, order=v_order
            )
            w = pcu.triangle_soup_fast_winding_number(
                vertices,
                faces,
                points_for_sdf,
            )
            mask = w > 0.99
            lookup_points = lookup_points[mask]
            index_points = index_points[mask]

        # do the actual point -> supervoxel lookup (this is the slow part)
        scatter_df = scattered_points(
            lookup_points, cv, client, root_id=root_id, timestamp=timestamp
        )
        scatter_df = scatter_df.reset_index()
        scatter_df["point_index"] = missing_point_ids[index_points]

        scatter_df["distance"] = np.linalg.norm(
            scatter_df.reset_index()[["x", "y", "z"]].values * res
            - closest_pts[scatter_df["point_index"]],
            axis=1,
        )

        valid_scatter_df = scatter_df[scatter_df["root_id"] == root_id]

        min_idx = valid_scatter_df.groupby(["point_index"])["distance"].idxmin()
        info.append(valid_scatter_df.loc[min_idx])

        # inject current mapping into mapping
        # TODO this should be a set
        point_mapping.update(min_idx.to_dict())
        mapping_keys = list(point_mapping.keys())
        missing_point_ids = np.setdiff1d(
            np.arange(len(cast_closest_points)), mapping_keys
        )

        # missing_points = cast_closest_points[missing_keys]

        distance += 1

    info = pd.concat(info)
    info = info.set_index("point_index").reindex(np.arange(len(closest_pts)))

    info["query_pt_x"] = points[:, 0]
    info["query_pt_y"] = points[:, 1]
    info["query_pt_z"] = points[:, 2]

    info["mesh_pt_x"] = closest_pts[:, 0]
    info["mesh_pt_y"] = closest_pts[:, 1]
    info["mesh_pt_z"] = closest_pts[:, 2]

    info["query_mesh_distance_nm"] = np.linalg.norm(
        info[["query_pt_x", "query_pt_y", "query_pt_z"]].values
        - info[["mesh_pt_x", "mesh_pt_y", "mesh_pt_z"]].values,
        axis=1,
    )

    info = info.rename(
        columns={
            "x": "voxel_pt_x",
            "y": "voxel_pt_y",
            "z": "voxel_pt_z",
            "distance": "mesh_voxel_distance_nm",
        }
    )

    info = info[
        [
            "query_pt_x",
            "query_pt_y",
            "query_pt_z",
            "mesh_pt_x",
            "mesh_pt_y",
            "mesh_pt_z",
            "voxel_pt_x",
            "voxel_pt_y",
            "voxel_pt_z",
            "supervoxel_id",
            "root_id",
            "query_mesh_distance_nm",
            "mesh_voxel_distance_nm",
        ]
    ]

    return info


def map_points(
    points: np.ndarray,
    root_id: int,
    client: CAVEclient,
    initial_distance: int = 0,
    max_distance: int = 4,
    verbose=False,
    *,
    point_offset=None,
    convention: str = "round",
    timestamp=None,
    lru_bytes: int = int(1e8),
) -> pd.DataFrame:
    """
    Map points in space to voxels and their corresponding supervoxels.

    Each point goes to a voxel (see `convention`). If that voxel is not part of
    `root_id`, the search grows one voxel shell at a time, up to `max_distance`,
    and keeps the nearest voxel of the root in the first shell that holds one.

    The chunkedgraph calls use the client's own retry settings. For long runs, raise
    them before building the client, for example
    `caveclient.set_session_defaults(max_retries=10, backoff_factor=1,
    backoff_max=240, status_forcelist=(429, 500, 502, 503, 504))`. The
    segmentation reads are retried by CloudFiles.

    Parameters
    ----------
    points :
        Array of points in space to be mapped. Must be of shape (n_points, 3), and
        are assumed to be in coordinates of nanometers.
    root_id :
        The root id of the object to map to.
    client :
        The CAVEclient object to use for mapping.
    initial_distance :
        The initial distance to search for a mapping. Default is 0. Units are in voxels.
    max_distance :
        The maximum distance to search for a mapping. Default is 4. Units are in voxels.
    verbose :
        Whether to print progress messages.
    point_offset :
        Nanometers added to every point before the voxel conversion. `None` adds
        nothing. `"mesh"` adds [mesh_point_offset][cave_mapper.mesh_point_offset],
        for points taken from the object's mesh, which sits half a mesh voxel off
        the voxel grid.
    convention :
        How a point becomes a voxel. `"floor"` is the voxel that holds the point,
        as CloudVolume and the CAVE materialization engine do it. `"round"` (the
        default, for backward compatibility) treats `i * res` as the voxel center.
    timestamp :
        The time for the `get_roots` calls. `None` uses the end of the root's
        lifetime (or now, for a latest root), so any time the root is valid gives
        the same answer.
    lru_bytes :
        Size of the CloudVolume chunk cache. Points close together share chunks.

    Returns
    -------
    :
        A DataFrame with one row per point, in the input order. A point with no
        voxel of the root within `max_distance` has nulls. Columns:

        - `query_pt_x`, `query_pt_y`, `query_pt_z`: The original points in space.
        - `voxel_pt_x`, `voxel_pt_y`, `voxel_pt_z`: The corresponding voxel coordinates.
            Units are the same as `client.chunkedgraph.base_resolution`.
        - `anchor_pt_x`, `anchor_pt_y`, `anchor_pt_z`: The voxel's minimum corner in
            nanometers, `voxel * resolution`. A floor lookup of this point, as the
            materialization engine does it, returns the same voxel at any
            annotation resolution that divides the segmentation resolution.
        - `supervoxel_id`: The supervoxel id of the corresponding voxel.
        - `root_id`: The root id of the object.
        - `query_voxel_distance_nm`: The distance from the offset point to the
           center of the chosen voxel (under `convention`).
        - `search_distance`: The shell the voxel was found in, in voxels. 0 means
           the point's own voxel.
    """
    cv: CloudVolume = client.info.segmentation_cloudvolume(
        progress=verbose, lru_bytes=lru_bytes
    )
    res = np.asarray(cv.meta.resolution(0), dtype=float)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    n_points = len(points)

    if point_offset is None:
        offset = np.zeros(3)
    elif isinstance(point_offset, str):
        if point_offset != "mesh":
            raise ValueError(
                f"point_offset must be None, 'mesh' or 3 numbers, not {point_offset!r}"
            )
        offset = mesh_point_offset(cv)
    else:
        offset = np.asarray(point_offset, dtype=float).reshape(3)
    query = points + offset
    voxels = points_to_voxels(query, res, convention)
    # Where a voxel's center sits, in voxels, under each convention.
    center_shift = 0.5 if convention == "floor" else 0.0

    if timestamp is None:
        timestamp = _root_timestamp(client, root_id)

    best_voxel = np.zeros((n_points, 3), dtype=np.int64)
    best_svid = np.zeros(n_points, dtype=np.int64)
    best_distance = np.full(n_points, np.nan)
    best_shell = np.full(n_points, -1, dtype=np.int64)
    remaining = np.arange(n_points)
    for distance in range(initial_distance, max_distance + 1):
        if len(remaining) == 0:
            break
        # The first pass looks at the whole cube; later passes only at the new shell,
        # since every voxel inside it already missed.
        if distance == initial_distance:
            offsets = construct_neighbor_elements(distance, distance, distance)
        else:
            offsets = shell_offsets(distance)
        candidates = (voxels[remaining][:, None, :] + offsets[None, :, :]).reshape(
            -1, 3
        )
        owner = np.repeat(remaining, len(offsets))

        # do the actual point -> supervoxel lookup (this is the slow part)
        lookup = cv.scattered_points(
            candidates, coord_resolution=res, agglomerate=False
        )
        svids = np.array([lookup[tuple(v)] for v in candidates], dtype=np.int64)
        unique_svids, inverse = np.unique(svids, return_inverse=True)
        unique_roots = np.zeros(len(unique_svids), dtype=np.int64)
        nonzero = unique_svids != 0
        if nonzero.any():
            unique_roots[nonzero] = client.chunkedgraph.get_roots(
                unique_svids[nonzero], timestamp=timestamp
            )
        valid = unique_roots[inverse.ravel()] == root_id

        dists = np.linalg.norm((candidates + center_shift) * res - query[owner], axis=1)
        dists[~valid] = np.inf
        order = np.lexsort((dists, owner))
        first = np.ones(len(order), dtype=bool)
        first[1:] = owner[order][1:] != owner[order][:-1]
        winners = order[first & np.isfinite(dists[order])]
        found = owner[winners]
        best_voxel[found] = candidates[winners]
        best_svid[found] = svids[winners]
        best_distance[found] = dists[winners]
        best_shell[found] = distance
        remaining = np.setdiff1d(remaining, found)

    found = best_shell >= 0

    def masked(values: np.ndarray) -> pd.arrays.IntegerArray:
        out = pd.array(values, dtype="Int64")
        out[~found] = pd.NA
        return out

    info = pd.DataFrame(
        {
            "query_pt_x": points[:, 0],
            "query_pt_y": points[:, 1],
            "query_pt_z": points[:, 2],
        }
    )
    for i, axis in enumerate("xyz"):
        info[f"voxel_pt_{axis}"] = masked(best_voxel[:, i])
    for i, axis in enumerate("xyz"):
        info[f"anchor_pt_{axis}"] = np.where(found, best_voxel[:, i] * res[i], np.nan)
    info["supervoxel_id"] = masked(best_svid)
    info["root_id"] = masked(np.full(n_points, root_id, dtype=np.int64))
    info["query_voxel_distance_nm"] = best_distance
    info["search_distance"] = masked(best_shell)
    return info

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


def cast_points_to_chunked(points: np.ndarray, cv: CloudVolume) -> np.ndarray:
    """
    Cast points to the chunked representation of the mesh.
    """
    points = np.array(points, dtype=np.float32)
    points_cast = points / np.array(cv.meta.resolution(0))
    points_cast = np.round(points_cast).astype(int)
    return points_cast


def scattered_points(
    points: np.ndarray,
    cv: CloudVolume,
    client: CAVEclient,
    root_id: Optional[int] = None,
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

    # TODO add a timestamp lookup here and pass it into get_roots
    if root_id is not None:
        timestamp = client.chunkedgraph.get_root_timestamps(root_id, latest=True)[0]
    else:
        timestamp = None

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
        scatter_df = scattered_points(lookup_points, cv, client, root_id=root_id)
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

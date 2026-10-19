# cave-mapper

Tools for mapping between objects and representations in the Connectome Annotation and
Versionining Engine (CAVE) ecosystem.

This package is a work in progress and is mainly a place to stash functions that I
find useful.

## Installation

```bash
pip install cave-mapper
```

or

```bash
uv pip install cave-mapper
```

or 

```bash
uv add install cave-mapper
```

## Available Functions

- `map_points`: This takes points, and a root ID (likely points annotated inside an object) and maps them to voxels/supervoxel IDs that are guaranteed to be part of the root ID. This is useful for mapping points to a specific object in a 3D space.

- `map_points_via_mesh`: This takes points and a root ID (likely points that were annotated on the surface of that object) and maps them to voxels/supervoxel IDs that are guaranteed to be part of the root ID.

## Mapping points from a mesh

For a point inside an object's mesh (for example, a point in a dendritic spine), use the mesh offset and the `floor` convention:

```python
from caveclient import CAVEclient, set_session_defaults
from cave_mapper import map_points

# Optional: more retries for the chunkedgraph calls than the client default. This applies
# to every client created after it, so call it before building the client.
set_session_defaults(
    max_retries=10,
    backoff_factor=1,
    backoff_max=240,
    status_forcelist=(429, 500, 502, 503, 504),
)
client = CAVEclient("minnie65_phase3_v1")
mapping = map_points(points, root_id, client, point_offset="mesh", convention="floor")
```

The `anchor_pt_x`, `anchor_pt_y` and `anchor_pt_z` columns are the chosen voxel's minimum corner in nanometers. The CAVE materialization engine looks up the same voxel for this point. Points deep inside the mesh map more reliably than points near its surface.

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

- `map_points_via_mesh`: This takes points and a root ID (likely points that were annotated on the surface of that object) and maps them to voxels/supervoxel IDs that are guaranteed to be part of the root ID.
# cf-utils

Miscellaneous small utilities. Nothing interesting here.

---

## `mesh.py` — NeoX `.cmesh` / `.mesh` parser

Parses the NeoX mesh container into positions / normals / tangents / UVs /
colours / skinning data plus a per-submesh breakdown. Pure Python, no
dependencies.

### Container layout

```
magic   u32   0xBBC88034
version u16   (2, 3, 4 and 7 are handled)
         u16   0x0500
has_bones u16 (1 / 4 => skinned)
         u16   0x0000
...skeleton block (if skinned)...
load_offset u32   -> seek here, then read the payload
```

The skeleton block is a bone count, that many parent indices (one byte per
index in version 4, otherwise 2 bytes for big skeletons), 32-byte name slots,
an optional block of 28 bytes per bone, 16 floats of inverse bind matrix per
bone and a tail byte that must be zero.

### The part table

Both payload layouts open with a table of part records: a vertex count, a face
count, a UV layer count and a colour flag. Two details make it awkward to read:

* **Records overlap.** Consecutive records start 10 bytes apart, so the fourth
  field of one record is the first two bytes of the next record's vertex count.
  They are not an array of fixed-size structs.
* **The table has no length.** There is no record count and no size field.

What terminates it depends on the version. Version 4 (and 7) carry this table
inline, so the mesh segments itself and needs no companion `.gim`; the walk stops
at the first record that is followed by the mesh's grand totals — which is how
those totals are learned in the first place:

```
Σ vertex_count == mesh.vertex_count  and  Σ face_count == mesh.face_count
```

Version 2/3 records instead end at a record whose fourth field is 1, with the
totals following it.

If the sums do **not** reconcile, the parser deliberately leaves `submeshes`
empty so the caller can fall back to a single-submesh path, rather than emitting
N zero-face submeshes (which renders as a blank screen). That check has to run
after the totals are stored, or it compares against zeros and never passes.

### Payload after the table

Version 4:

```
u8 fmt             bit 0 = raw f32 positions;
                   otherwise 3 x 20-bit fixed point packed into 8 bytes/vertex
                   bit 1 = absolute u16 indices (else delta encoded)
                   bit 2 = no tangent stream
positions          f32 x3, or the packed form above
normals            i8 x3, divided by 127
tangents           i8 x3 plus a 3-bits-per-vertex widen mask
indices            u16 x3
UVs                f16 x2 per declared layer, part by part
colours            u8 x4, for the parts that declared one
skinning           4 x u8 joints, 4 x u16 weights (/65535), 32-byte bone bitmap
```

Version 2/3 have no format byte: the payload is a flat stream whose layout is
inferred. Which attribute streams are present is not tagged per stream — it is
worked out from how many bytes are left over for the declared vertex and face
counts, by `detect_layout()`. The sizes it matches against are a position (24 B),
a face (6 B), a UV pair (8 B, or 4 B as f16), a normal (12 B), a colour (4 B), a
tangent (12 B), skinning (20 B) and two smaller extras of 4 B and 2 B. Because
it is an inference, a wrong face count or skeleton flag shows up as a
wrong-but-plausible attribute set rather than as a clean error.

## `gim.py` — NeoX `.gim`

XML side-car: overall vertex/face counts, bounding box, the submesh list
(name / material index / counts when present) and referenced GIS paths.
Note that many `.gim` files carry **no** per-submesh counts; see above for where
v4 gets them instead.

## Requirements

None. `mesh.py` is pure standard library.

## Usage

```python
from mesh import parse_mesh
from gim import parse_gim_file

mesh = parse_mesh(open("model.cmesh", "rb").read())
print(mesh.version, mesh.vertex_count, mesh.face_count, len(mesh.submeshes))

gim = parse_gim_file("model.gim")
for i, s in enumerate(gim.submeshes or []):
    print(i, s.name, s.mtl_idx)
```

---

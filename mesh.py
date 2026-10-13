# -*- coding: utf-8 -*-
"""NeoX mesh container reader: ``.cmesh`` / ``.mesh`` -> plain arrays.

Scope
-----
One entry point, :func:`parse_mesh`, turns a NeoX mesh blob into positions,
normals, tangents, per-layer UVs, vertex colours, skin weights, triangles and a
per-submesh breakdown, plus the skeleton block when the file has one.

Container
---------
Little-endian throughout; the header is a fixed 12 bytes::

    u32  magic            0xBBC88034 -- anything else is not a mesh
    u16  version          versions seen in shipping data: 2, 3, 4, 7
    u16  0x0500           constant
    u16  bone_mode        0 and 1 have no skeleton, 3 and 4 do
    u16  0x0000           constant

A ``bone_mode`` of 1 or 4 means a skeleton block sits between the header and the
load offset, laid out as::

    u16       bone_count
    u16|u8    bone_count parent indices      (all bits set == "this is a root")
    u8[32]    bone_count name slots, NUL padded
    u8        extra flag; non-zero => bone_count * 28 more bytes, skipped
    f32[16]   bone_count inverse bind matrices, row major
    u8        tail flag, must be 0

The width of the parent index column is not recorded anywhere; version 4 always
uses one byte, and older files are resolved by :func:`_parent_index_width`,
which rejects a width whose name slots do not look like NUL-padded text.

The header ends with::

    u32       load_offset   absolute file offset of the geometry payload

Everything from ``load_offset`` on is payload.  Which payload layout applies is
decided by ``version``, not by any flag: version 4 uses the inline layout below,
anything else the flat one.

The part table
--------------
Both layouts open with a table of part records describing how the mesh is split.
A record carries a vertex count, a face count, a UV layer count and a colour
flag.  It occupies 12 bytes::

    u32   vertex_count
    u32   face_count
    u8    uv_layers
    u8    has_color
    u16   fourth field -- 1 on the terminating record, 0 otherwise

Records are *not* laid out back to back: each one starts 10 bytes after the
previous one, so consecutive records overlap by two bytes and a record's fourth
field is the first two bytes of the next record's vertex count.  That overlap is
why the table cannot be read as an array of structs -- :func:`_read_part_table`
walks it through explicit byte windows instead -- and it is also why the walk
cannot simply trust that field.

The table has no record count and no length field.  What terminates it differs
by version:

* version 4 -- the terminating record is followed by the mesh totals as two
  ``u32`` values.  The walk stops at the first record whose following value
  equals the running vertex sum (the face sum is then checked as well), which is
  how the totals are discovered in the first place.  A table that never
  reconciles is an error, not a fallback.
* version 2/3 -- the terminating record's fourth field is 1, and the totals
  follow it.  The geometry is a flat stream, so the part split comes from the
  records here (with the companion ``.gim`` describing names and materials).

After the totals comes one format byte (version 4 only)::

    bit 0   0 -> positions are 3 x 20-bit fixed point packed into 8 bytes
            1 -> positions are 3 x f32, 12 bytes per vertex
    bit 1   0 -> indices are delta encoded (u16, running sum, wrapping)
            1 -> indices are absolute u16
    bit 2   0 -> a tangent stream is present
            1 -> it is not

Position channels are packed low-to-high as 12 fraction bits (scale 1/2048),
7 integer bits and a sign bit.  After the format byte the version-4 streams
appear in a fixed order: positions, normals (3 x i8, divided by 127), tangents
(3 x i8 plus a 3-bits-per-vertex widen mask), indices, UVs (f16 x 2 per declared
layer, laid out part by part), colours (u8 x 4 for the parts that declared one)
and finally skinning (4 x u8 joints, 4 x u16 weights scaled by 1/65535, and a
bone bitmap that is skipped).

The version-2/3 payload in turn is a flat stream whose attribute layout is not
tagged per stream.  It is inferred from how many bytes are left over for the
declared vertex and face counts, by :func:`detect_layout`; the fixed sizes
involved are a position (24 B), a face (6 B), a UV pair (8 B, or 4 B as f16), a
normal (12 B), a colour (4 B), a tangent (12 B), skinning (20 B: 4 joints + 4
floats) and two smaller extras of 4 B and 2 B.  Because the layout is a guess, a
wrong face count or a wrong skeleton flag shows up as a plausible-looking but
wrong attribute set rather than as a clean error.

Submeshes
---------
Submeshes are reported from the part records when -- and only when -- the
recorded counts reconcile with the mesh totals.  When they do not, the list is
left empty on purpose: callers then fall back to treating the mesh as a single
submesh.  Emitting submeshes that do not add up produces zero-face draws, which
render as an empty screen, so an empty list is the safe answer.  The check
therefore has to run *after* the totals are stored in the result.

本项目的实测记录（本项目自己的工作，不是从别处抄来的）
-------------------------------------------------------
* v4 自带整张子网表，**不依赖同名 `.gim`**。实测一份 1,556,547 B 的 v4 模型：
  表在文件偏移 30895 起、15 段、Σvc=41277、Σfc=54564，与网格总数**逐项精确相符**，
  每段的 `vertex_start` 由前缀和自然连续无缝。另一份 1,130,341 B 的 v4 模型是
  16 段、Σvc=31524、Σfc=41292。两份都覆盖 uv_layers 1~2 与 has_color 0/1。
* ⚠ **记录是 12 字节、相邻记录只差 10 字节**（`u32 vc | u32 fc | u8 uv | u8 color |
  u16`），所以第 4 个字段正好是下一条记录 vc 的低 16 位。本项目旧笔记写成
  "v4 表是 10 字节记录、u16 标志落在 +8"，照那个写守卫会恒假；本轮用
  `trace_ref2.py` 复刻参考实现的游标轨迹才坐实（实测 00129_02 的 15 段落在
  30895/30905/.../31035，终止记录 @31035 后紧跟 (41277, 54564)）。
* ⛔ **计数不符时绝不能把子网表写进 `submeshes`。** 曾经踩过的已知回归：
  上层在 `submeshes` 为空时会回退成"单个子网"渲染，反之若给它 N 个**计数为 0**
  的子网，就会画 N 个 0 面 draw ⇒ **整屏空白**（判据 E：主体占屏比 0.8778 → 0.0259）。
  所以这里的守卫宁可返回空表。
* ⚠ **守卫必须在总数赋值之后**执行（见 `_record_geometry` 的调用点）。
  先前把它放在赋值之前，比较的是 0，守卫恒假、静默失效，且不报任何错。
* 骨架父索引列宽（1B/2B）文件里没有记录。实测判据：**骨数 ≥ 256 用 2 字节**，
  小于则 1 字节；配合"名字槽必须是 NUL 填充文本"的校验可以双向确认
  （实测 182 骨的模型按 1 字节读时名字槽全部合法，按 2 字节读则第一个槽已是乱码）。
  多根骨架会补一个 `dummy_root` 并把它作为唯一根。
* 头部的 end-address 对 v2/v3 **是必需的**：属性流恰好铺到那里，字节预算必须量到它。
  早先误用了文件尾附近的另一个偏移字段，预算偏小 ⇒ `detect_layout` 判定失败
  ⇒ legacy 的 `format_type` 变成 -99（应为 1）。总数与几何本身不受影响。
* 顶点属性组合是**反推**出来的（文件不逐项标注）。反推错的表现是"属性集不对"，
  而不是干净报错。
* v4 的 `fmt` 字节：bit0=1 位置为裸 f32×3，否则 3×20 位定点（每通道低 12 位小数
  /2048、次 7 位整数、最高位符号）；bit1=1 为绝对 u16 索引，否则增量编码；
  bit2=0 表示存在切线流。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

MAGIC = 0xBBC88034
HEADER_CONSTANT = 0x0500
SKINNED_MODES = (1, 4)
BONE_COUNT_16BIT_MIN = 256
NAME_SLOT_SIZE = 32
SKELETON_EXTRA_BYTES = 28
ROOT_INDEX = -1
MAX_PART_RECORDS = 256
PART_STRIDE = 10                # distance between the starts of two records
PART_TAIL_OFFSET = 10           # record field 4; also where the totals follow

POS_FLOATS_PER_VERTEX = 24      # bytes of one position when stored as f32
FACE_BYTES = 6                  # bytes of one triangle (3 x u16)
UV_BYTES = 8                    # bytes of one UV pair when stored as f32
UV_HALF_BYTES = 4               # ... and as f16
NORMAL_BYTES = 12               # bytes of one normal when stored as f32
COLOR_BYTES = 4                 # bytes of one RGBA colour
SKIN_BYTES = 20                 # bytes of one skin record: 4 x u8 + 4 x f32
SMALL_EXTRA_BYTES = 4
TINY_EXTRA_BYTES = 2
BONE_BITMAP_BYTES = 32

POS_FIXED_FRAC_BITS = 12
POS_FIXED_FRAC_MASK = (1 << POS_FIXED_FRAC_BITS) - 1
POS_FIXED_FRAC_SCALE = 2048.0
POS_FIXED_INT_BITS = 7
POS_FIXED_INT_MASK = (1 << POS_FIXED_INT_BITS) - 1
POS_SHIFT_PER_CHANNEL = 20
POS_CHANNEL_BITS = 20
POS_CHANNEL_MASK = (1 << POS_CHANNEL_BITS) - 1

LAYOUT_ABSENT = -99             # payload matches no known attribute set
LAYOUT_UNSUPPORTED = -1
LAYOUT_V7_F16_POS = 100         # internal tag, not a format number
QUANT_POS_RANGE = ((-3800.0, 3800.0), (-3500.0, 3500.0), (-6000.0, 6000.0))
QUANT_NORMAL_RANGE = (-1.0, 1.0)
RESCALE_LAYOUTS = (20, 21)
F16_FLOAT_LAYOUTS = (LAYOUT_V7_F16_POS, 4, 5)
WIDE_JOINT_LAYOUTS = (2, 3, 5)
SKIP_SIX_BYTE_LAYOUTS = (4, 5, 7, LAYOUT_V7_F16_POS)


# ---------------------------------------------------------------------------
# result types
# ---------------------------------------------------------------------------

@dataclass
class SubmeshInfo:
    """One drawable slice of a mesh: a vertex range and a triangle range."""
    index: int = 0
    name: str = ""
    mtl_idx: int = 0
    face_start: int = 0
    face_count: int = 0
    vertex_count: int = 0
    vertex_start: int = 0


@dataclass
class MeshData:
    """Everything :func:`parse_mesh` recovers from one mesh file."""
    version: int = 0
    bone_mode: int = 0
    bone_names: list = field(default_factory=list)
    bone_parents: list = field(default_factory=list)
    bone_matrices: list = field(default_factory=list)
    positions: list = field(default_factory=list)
    colors: list = field(default_factory=list)
    normals: list = field(default_factory=list)
    tangents: list = field(default_factory=list)
    faces: list = field(default_factory=list)
    uv: list = field(default_factory=list)
    joints: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    vertex_count: int = 0
    face_count: int = 0
    format_type: int = LAYOUT_ABSENT
    submeshes: list = field(default_factory=list)


class _PartRecord:
    """The four counts one part record declares, plus where it started."""
    __slots__ = ("vertex_count", "face_count", "uv_layers", "has_color", "at")

    def __init__(self, vertex_count, face_count, uv_layers, has_color, at):
        self.vertex_count = vertex_count
        self.face_count = face_count
        self.uv_layers = uv_layers
        self.has_color = has_color
        self.at = at


# ---------------------------------------------------------------------------
# byte access
# ---------------------------------------------------------------------------

class _MeshReader:
    """Little-endian cursor over the mesh blob.

    Mirrors the file's own structure: the caller walks the skeleton, jumps to
    the load offset, then reads the payload in declaration order.
    """

    def __init__(self, blob: bytes):
        self.blob = blob
        self.at = 0

    def skip(self, count: int) -> None:
        self.at += count

    def jump(self, offset: int) -> None:
        self.at = offset

    def take(self, count: int) -> bytes:
        chunk = self.blob[self.at:self.at + count]
        self.at += count
        return chunk

    def u8(self) -> int:
        value = struct.unpack_from("<B", self.blob, self.at)[0]
        self.at += 1
        return value

    def u16(self) -> int:
        value = struct.unpack_from("<H", self.blob, self.at)[0]
        self.at += 2
        return value

    def u16_at(self, offset: int) -> int:
        return struct.unpack_from("<H", self.blob, offset)[0]

    def u32(self) -> int:
        value = struct.unpack_from("<I", self.blob, self.at)[0]
        self.at += 4
        return value

    def i16(self) -> int:
        value = struct.unpack_from("<h", self.blob, self.at)[0]
        self.at += 2
        return value

    def pair32_at(self, offset: int):
        return struct.unpack_from("<II", self.blob, offset)

    def u64(self) -> int:
        value = struct.unpack_from("<Q", self.blob, self.at)[0]
        self.at += 8
        return value

    def f32(self) -> float:
        value = struct.unpack_from("<f", self.blob, self.at)[0]
        self.at += 4
        return value

    def f16(self) -> float:
        value = struct.unpack_from("<e", self.blob, self.at)[0]
        self.at += 2
        return float(value)

    def signed_column(self, count: int, divide_by: float = 127.0):
        """Read ``count`` signed bytes as floats."""
        return [b / divide_by if b < 128 else (b - 256) / divide_by
                for b in self.take(count)]


def _rows(flat, width: int, count: int):
    """Cut a flat scalar list into ``count`` rows of ``width``."""
    return [tuple(flat[i * width:(i + 1) * width]) for i in range(count)]


def _bits(raw: bytes, count: int):
    """Expand little-endian bit-packed flags into a list of booleans."""
    return [bool(raw[i >> 3] & (1 << (i & 7))) for i in range(count)]


# ---------------------------------------------------------------------------
# skeleton
# ---------------------------------------------------------------------------

def _slot_is_text(slot: bytes) -> bool:
    """True when a 32-byte skeleton slot looks like a NUL-terminated name."""
    if len(slot) != NAME_SLOT_SIZE or b"\x00" not in slot:
        return False
    head, tail = slot.split(b"\x00", 1)
    if tail.strip(b"\x00"):
        return False
    if not head:
        return True
    try:
        head.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _parent_index_width(blob: bytes, at: int, bone_count: int) -> int:
    """Decide whether parent indices are 1 or 2 bytes wide.

    Nothing in the file states the width, so it is recovered from the layout it
    implies.  The narrow reading is tried first -- it is the one every skeleton
    under :data:`BONE_COUNT_16BIT_MIN` bones uses -- and is accepted only if the
    name slots that follow it all look like NUL-padded text.  A wrong width puts
    the name block at the wrong offset, which shows up as non-text slots almost
    immediately, so the check is decisive in practice.
    """
    if bone_count >= BONE_COUNT_16BIT_MIN:
        return 2
    probe = at + bone_count
    if probe + NAME_SLOT_SIZE > len(blob):
        return 2
    for i in range(min(bone_count, 8)):
        slot = blob[probe + i * NAME_SLOT_SIZE:probe + (i + 1) * NAME_SLOT_SIZE]
        if not _slot_is_text(slot):
            return 2
    return 1


def _slot_text(slot: bytes) -> str:
    """Decode one 32-byte name slot, NUL padding stripped."""
    return slot.split(b"\x00", 1)[0].decode("utf-8", "replace")


def _read_skeleton(r: _MeshReader, out: MeshData) -> None:
    """Fill the bone tables, adding a synthetic root when there are several."""
    bone_count = r.u16()
    # Version 4 stores parent indices as bytes regardless of bone count; older
    # files have to be probed.
    if out.version == 4:
        width = 1
    else:
        width = _parent_index_width(r.blob, r.at, bone_count)
    root_marker = 0xFFFF if width == 2 else 0xFF

    parents = []
    for _ in range(bone_count):
        parent = r.u16() if width == 2 else r.u8()
        parents.append(ROOT_INDEX if parent == root_marker else parent)
    names = [_slot_text(r.take(NAME_SLOT_SIZE)) for _ in range(bone_count)]

    if r.u8():
        r.skip(SKELETON_EXTRA_BYTES * bone_count)

    matrices = [
        [list(struct.unpack_from("<4f", r.take(16))) for _ in range(4)]
        for _ in range(bone_count)
    ]

    tail = r.u8()
    if tail != 0:
        raise ValueError(
            "unexpected skeleton tail flag %d at %#x" % (tail, r.at - 1))

    # A mesh may declare several root bones.  Reparent them under one synthetic
    # root so the result is a tree with a single entry point.
    if parents.count(ROOT_INDEX) > 1:
        synthetic = len(parents)
        parents = [synthetic if p == ROOT_INDEX else p for p in parents] + [ROOT_INDEX]
        names = names + ["dummy_root"]
        matrices = matrices + [[[1.0 if i == j else 0.0 for j in range(4)]
                                for i in range(4)]]

    out.bone_parents = parents
    out.bone_names = names
    out.bone_matrices = matrices


# ---------------------------------------------------------------------------
# part table
# ---------------------------------------------------------------------------

def _read_part_table(r: _MeshReader, by_totals: bool):
    """Walk the part table; return its records, the mesh totals and the offset
    where the totals end (i.e. where a version-4 format byte follows).

    Records are read through windows at :data:`PART_STRIDE` byte intervals, so
    each window overlaps the next by eight bytes: the record's fourth field is
    the next record's vertex count.  Reading is done from absolute addresses
    rather than by seeking, because that overlap makes the fields of one record
    part of the address arithmetic of the next.

    ``by_totals`` selects the version-4 stop rule (the running sums are echoed
    after the last record) from the version-2/3 one (the fourth field is 1).
    """
    base = r.at
    at = base
    parts = []
    while True:
        vertex_count, face_count = r.pair32_at(at)
        uv_layers = r.blob[at + 8]
        has_color = r.blob[at + 9]
        parts.append(_PartRecord(vertex_count, face_count, uv_layers, has_color, at))

        tail = r.u16_at(at + PART_TAIL_OFFSET)
        total_v, total_f = r.pair32_at(at + PART_TAIL_OFFSET + TINY_EXTRA_BYTES)
        if by_totals:
            if tail and total_v == sum(p.vertex_count for p in parts) \
                    and total_f == sum(p.face_count for p in parts):
                r.jump(at + PART_TAIL_OFFSET + TINY_EXTRA_BYTES + 8)
                return parts, total_v, total_f
        elif tail == 1:
            r.jump(at + PART_TAIL_OFFSET + TINY_EXTRA_BYTES + 8)
            return parts, total_v, total_f

        at += PART_STRIDE
        if len(parts) > MAX_PART_RECORDS:
            raise ValueError("part table does not terminate")


# ---------------------------------------------------------------------------
# attribute layout inference (version 2/3)
# ---------------------------------------------------------------------------

def detect_layout(tail_bytes, vertex_count, face_count, uv_floats,
                  bone_mode, version):
    """Identify the attribute set from the byte budget of the payload.

    Counts the declared streams against ``tail_bytes`` -- the payload bytes
    left once the part table and totals have been consumed -- and returns
    ``(tag, lead_in)``: the tag names the attribute set, ``lead_in`` is a count
    of bytes to drop before positions start.

    ``uv_floats`` is the number of scalar UV components the part records
    declare, i.e. UV pairs times layers; UV components are 8 bytes as f32.

    Tags 1/2/3 are the plain (position + normal) families, 4/5 the f16 position
    families, 20/21 quantised positions that get a second de-quantisation pass,
    22/23 the same with f16 UVs, and :data:`LAYOUT_V7_F16_POS` the v7 shape
    handled with f16 UVs.  :data:`LAYOUT_ABSENT` means nothing fitted the
    budget.
    """
    budget = tail_bytes
    uv_bytes = uv_floats * UV_BYTES
    if bone_mode in SKINNED_MODES:
        budget -= SKIN_BYTES * vertex_count

    if bone_mode in (0, 1):
        # Probe stream by stream, subtracting the size of each candidate stream
        # from what is left over.  A remainder of 0 (or exactly one name slot,
        # which is how files with a little padding show up) identifies the set.
        left = (budget - vertex_count * POS_FLOATS_PER_VERTEX
                - face_count * FACE_BYTES - uv_bytes - TINY_EXTRA_BYTES)
        if left == 0 or left == NAME_SLOT_SIZE:
            return 1, 0
        left -= vertex_count * NORMAL_BYTES
        if left == 0 or left == NAME_SLOT_SIZE:
            return 1, 0
        left -= vertex_count * COLOR_BYTES
        if 0 < left < vertex_count - 1:
            return 2, 0
        left -= vertex_count * SMALL_EXTRA_BYTES
        if 0 < left < vertex_count - 1:
            return 3, 0
        if version == 4:
            return LAYOUT_UNSUPPORTED, 0
        return LAYOUT_ABSENT, 0

    if bone_mode not in (3, 4):
        return -98, 0

    left = (budget - vertex_count * NORMAL_BYTES - face_count * FACE_BYTES
            - uv_bytes - TINY_EXTRA_BYTES)
    for tag, step in ((4, 0), (5, SMALL_EXTRA_BYTES), (4, TINY_EXTRA_BYTES),
                      (5, SMALL_EXTRA_BYTES)):
        if left in (0, NAME_SLOT_SIZE):
            return tag, 0
        left -= vertex_count * step

    if version == 7:
        for uv_divisor, tag in ((uv_bytes // 2, 22), (uv_bytes, 20)):
            left = (budget - vertex_count * SKIN_BYTES - face_count * FACE_BYTES
                    - uv_divisor)
            if 0 <= left < vertex_count - 1:
                return tag, left
            left -= vertex_count * SMALL_EXTRA_BYTES
            if 0 <= left < vertex_count - 1:
                return tag + 1, left
    return LAYOUT_V7_F16_POS, 0


def _dequantise(value: int, lo: float, hi: float) -> float:
    """Map a signed 16-bit code onto [lo, hi]."""
    return lo + ((value + 32768) / 65535.0) * (hi - lo)


# ---------------------------------------------------------------------------
# the pieces both payload layouts share
# ---------------------------------------------------------------------------

def _record_geometry(out: MeshData, vertex_count: int, face_count: int,
                     parts) -> None:
    """Store the totals and the submesh ranges, but only if they reconcile.

    The guard is the point of this function.  Callers fall back to a
    single-submesh draw when ``submeshes`` is empty, whereas a list whose counts
    do not add up produces zero-face draws and a blank screen.  It must run
    *after* the totals are stored, or the comparison is against zeros and the
    guard silently never passes.
    """
    if not parts:
        return
    if sum(p.vertex_count for p in parts) != vertex_count:
        return
    if sum(p.face_count for p in parts) != face_count:
        return
    vertex_start = 0
    face_start = 0
    for index, part in enumerate(parts):
        out.submeshes.append(SubmeshInfo(
            index=index,
            name="",
            mtl_idx=index,
            face_start=face_start,
            face_count=part.face_count,
            vertex_count=part.vertex_count,
            vertex_start=vertex_start,
        ))
        vertex_start += part.vertex_count
        face_start += part.face_count


def _read_colors(r: _MeshReader, parts, total_vertices: int):
    """Read the RGBA blocks, one per part that declared a colour stream."""
    colors = [(1.0, 1.0, 1.0, 1.0)] * total_vertices
    base = 0
    for part in parts:
        if part.has_color:
            for i in range(part.vertex_count):
                at = base + i
                rgba = r.take(4)
                if at < total_vertices:
                    colors[at] = (rgba[0] / 255.0, rgba[1] / 255.0,
                                  rgba[2] / 255.0, rgba[3] / 255.0)
        base += part.vertex_count
    return colors


def _read_uv_layers(r: _MeshReader, parts, total_vertices: int, half: bool):
    """Read the UV blocks, part by part and layer by layer.  Layer 0 wins."""
    layer_count = max((p.uv_layers for p in parts), default=0)
    if not layer_count:
        return []
    layers = [[0.0] * (total_vertices * 2) for _ in range(layer_count)]
    base = 0
    for part in parts:
        for layer in range(part.uv_layers):
            row = layers[layer]
            for i in range(part.vertex_count):
                u = r.f16() if half else r.f32()
                v = r.f16() if half else r.f32()
                at = (base + i) * 2
                row[at] = u
                row[at + 1] = v
        base += part.vertex_count
    return _rows(layers[0], 2, total_vertices)


# ---------------------------------------------------------------------------
# version-4 payload
# ---------------------------------------------------------------------------

def _read_v4_payload(r: _MeshReader, out: MeshData) -> None:
    """Read a version-4 payload: part table, totals, format byte, streams."""
    parts, total_v, total_f = _read_part_table(r, by_totals=True)
    fmt = r.u8()

    out.format_type = 30
    out.vertex_count = total_v
    out.face_count = total_f
    _record_geometry(out, total_v, total_f, parts)

    positions = []
    for _ in range(total_v):
        if fmt & 1:
            positions.append((r.f32(), r.f32(), r.f32()))
        else:
            packed = r.u64()
            channels = []
            for channel in range(3):
                code = (packed >> (POS_SHIFT_PER_CHANNEL * channel)) & POS_CHANNEL_MASK
                value = ((code & POS_FIXED_FRAC_MASK) / POS_FIXED_FRAC_SCALE
                         + ((code >> POS_FIXED_FRAC_BITS) & POS_FIXED_INT_MASK))
                if code >> (POS_FIXED_FRAC_BITS + POS_FIXED_INT_BITS):
                    value = -value
                channels.append(value)
            positions.append(tuple(channels))
    out.positions = positions

    out.normals = _rows(r.signed_column(total_v * 3), 3, total_v)

    if not (fmt >> 2) & 1:
        widths = r.signed_column(total_v * 3)
        mask = r.take(total_v * 3 // 8 + 1)
        for i, widen in enumerate(_bits(mask, total_v * 3)):
            if widen:
                widths[i] *= 3.0
        out.tangents = _rows(widths, 3, total_v)

    flat = struct.unpack_from("<%dH" % (total_f * 3), r.take(total_f * FACE_BYTES))
    if (fmt >> 1) & 1:
        indices = list(flat)
    else:
        indices = []
        running = 0
        for delta in flat:
            running = (running + delta) & 0xFFFF
            indices.append(running)
    out.faces = _rows(indices, 3, total_f)

    uv = _read_uv_layers(r, parts, total_v, half=True)
    out.uv = uv if uv else [(0.0, 0.0)] * total_v

    out.colors = _read_colors(r, parts, total_v)

    if out.bone_mode in SKINNED_MODES:
        joints = r.take(total_v * 4)
        out.joints = [tuple(joints[i * 4:(i + 1) * 4]) for i in range(total_v)]
        raw = struct.unpack_from("<%dH" % (total_v * 4), r.take(total_v * 8))
        out.weights = _rows([w / 65535.0 for w in raw], 4, total_v)
        r.skip(BONE_BITMAP_BYTES)


# ---------------------------------------------------------------------------
# version-2/3 payload
# ---------------------------------------------------------------------------

def _read_legacy_payload(r: _MeshReader, out: MeshData, end_offset: int) -> None:
    """Read a version-2/3 payload: flat streams, sized by a byte budget.

    ``end_offset`` is the address the header points at (the same field the side
    tables are read from).  The payload's attribute streams reach exactly up to
    it, so it is what the byte budget in :func:`detect_layout` is measured
    against -- not the file length and not any other offset.
    """
    parts, total_v, total_f = _read_part_table(r, by_totals=False)
    payload_at = r.at

    uv_floats = sum(p.vertex_count * p.uv_layers for p in parts)
    if total_v * 2 == uv_floats and parts[-1].uv_layers == 1:
        uv_floats = total_v

    tail_bytes = end_offset - payload_at
    color_bytes = sum(p.vertex_count * COLOR_BYTES for p in parts if p.has_color)
    layout, lead_in = detect_layout(tail_bytes - color_bytes, total_v, total_f,
                                    uv_floats, out.bone_mode, out.version)

    out.format_type = layout
    out.vertex_count = total_v
    out.face_count = total_f
    _record_geometry(out, total_v, total_f, parts)

    if layout == LAYOUT_UNSUPPORTED:
        raise NotImplementedError("legacy mesh layout %d is not supported" % layout)

    quantised = layout in (20, 21, 22, 23)
    halves = layout in F16_FLOAT_LAYOUTS
    wide_joints = layout in WIDE_JOINT_LAYOUTS

    def scalar():
        return r.f16() if halves else r.f32()

    if quantised:
        r.skip(lead_in)
        flat = []
        for _ in range(total_v):
            for lo, hi in QUANT_POS_RANGE:
                flat.append(_dequantise(r.i16(), lo, hi))
        out.positions = _rows(flat, 3, total_v)
        flat = []
        for _ in range(total_v):
            for _ in range(3):
                flat.append(_dequantise(r.i16(), *QUANT_NORMAL_RANGE))
        out.normals = _rows(flat, 3, total_v)
    else:
        out.positions = _rows([scalar() for _ in range(total_v * 3)], 3, total_v)
        out.normals = _rows([scalar() for _ in range(total_v * 3)], 3, total_v)

    flag = r.u16()
    if flag == 1 and layout in SKIP_SIX_BYTE_LAYOUTS:
        r.skip(total_v * 6)
    elif quantised:
        r.skip(total_v * 8 - TINY_EXTRA_BYTES)
    elif flag == 1:
        r.skip(total_v * NORMAL_BYTES)
    elif flag > 1:
        r.skip(flag * SMALL_EXTRA_BYTES)

    out.faces = _rows([r.u16() for _ in range(total_f * 3)], 3, total_f)

    if layout == 21:
        r.skip(total_v * SMALL_EXTRA_BYTES)
    if layout == 23:
        uv_floats //= 2

    if uv_floats and layout != LAYOUT_V7_F16_POS:
        if out.version in (2, 3):
            uv = _read_uv_layers(r, parts, total_v, half=False)
            out.uv = uv if uv else [(0.0, 0.0)] * total_v
        else:
            out.uv = _rows([r.f32() for _ in range(uv_floats * 2)], 2, uv_floats)
    else:
        out.uv = [(0.0, 0.0)] * total_v

    out.colors = _read_colors(r, parts, total_v)

    if out.bone_mode in SKINNED_MODES and layout != LAYOUT_V7_F16_POS:
        if wide_joints:
            out.joints = [tuple(r.u16() for _ in range(4)) for _ in range(total_v)]
        else:
            out.joints = [tuple(r.u8() for _ in range(4)) for _ in range(total_v)]
        bone_count = len(out.bone_names)
        out.weights = [
            _normalise_weights(joints, [r.f32() for _ in range(4)], bone_count)
            for joints in out.joints
        ]
    elif out.bone_mode in SKINNED_MODES:
        out.joints = [(0, 0, 0, 0)] * total_v
        out.weights = [(0.0, 0.0, 0.0, 0.0)] * total_v

    if layout in RESCALE_LAYOUTS and out.positions:
        out.positions = _rescale_positions(out.positions)


def _normalise_weights(joints, weights, bone_count):
    """Drop weights that are not finite or name no bone, then normalise.

    Unused slots hold NaN sentinels rather than zeros, so a row is only usable
    once those are cleared; rows with nothing left stay all-zero instead of
    becoming NaN through the division.
    """
    cleaned = [w if (w == w and w not in (float("inf"), float("-inf"))
                     and joints[i] < bone_count) else 0.0
               for i, w in enumerate(weights)]
    total = sum(cleaned)
    if total > 0:
        cleaned = [w / total for w in cleaned]
    return tuple(cleaned)


def _rescale_positions(positions):
    """Second pass for the quantised layouts: remap onto the observed bounds."""
    xs, ys, zs = zip(*positions)
    bounds = ((min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs)))
    return [
        tuple(_dequantise(int(v), bounds[k][0], bounds[k][1])
              for k, v in enumerate(p))
        for p in positions
    ]


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def parse_mesh(data: bytes) -> MeshData:
    """Parse a NeoX ``.cmesh`` / ``.mesh`` blob."""
    r = _MeshReader(bytes(data))
    out = MeshData()

    magic = r.u32()
    if magic != MAGIC:
        raise ValueError("not a NeoX mesh: magic=%#010x" % magic)
    out.version = r.u16()
    r.u16()                                     # constant header field
    out.bone_mode = r.u16()
    r.u16()                                     # constant header field

    if out.bone_mode in SKINNED_MODES:
        _read_skeleton(r, out)

    # Address of the side-table block; for version-2/3 files it also marks the
    # end of the payload's attribute streams.
    end_offset = r.u32()

    r.jump(end_offset)
    side_table_count = r.u16()
    offsets = [r.u32() for _ in range(side_table_count)]
    main_offset = offsets[0]

    # The offsets after the first point at side tables, not at geometry.  They
    # are not read as part of the mesh; their size only matters because the
    # payload byte budget of a version-2/3 file has to account for them.
    for offset in offsets[1:]:
        r.jump(offset)
        count = r.u32()
        r.skip(count * 12)
        count = r.u32()
        r.skip(count * 2)
        count = r.u32()
        r.skip(count * 4)

    r.jump(main_offset)

    if out.version == 4:
        _read_v4_payload(r, out)
    else:
        _read_legacy_payload(r, out, end_offset)
    return out

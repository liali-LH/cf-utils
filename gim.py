"""NeoX .gim 文件解析（网格索引/描述文件）。

.gim 是 XML 格式的网格元数据文件，包含：
- 整体网格统计（顶点数、面数、包围盒）
- Submesh 列表（名称、材质索引、顶点/面数量）
- 引用的 GIS 文件路径
- Socket 挂接点

.gim 本身不存几何数据，实际数据在 .gis/.gisacl 里。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SubmeshInfo:
    """单个子网格的描述信息。"""
    index: int                 # submesh 序号（0-based）
    name: str                  # 子网格名称
    mtl_idx: int               # 材质索引（对应 .mtg 的 Material_i）
    vertex_count: int          # 顶点数
    triangle_count: int        # 三角面数
    vertex_start: int = 0      # 在全局顶点数组中的起始索引（解析后计算）
    face_start: int = 0        # 在全局面数组中的起始索引（解析后计算）
    bounding_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounding_half: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class SocketInfo:
    """挂接点信息。"""
    name: str
    bind_type: int
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]  # 四元数
    bound_object: str = ""   # 绑定的对象路径


@dataclass
class GimData:
    """GIM 文件解析结果。"""
    vertex_count: int = 0
    triangle_count: int = 0
    tangent_enable: bool = True
    bounding_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounding_half: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounding_radius: float = 0.0
    submeshes: list[SubmeshInfo] = field(default_factory=list)
    gis_files: list[str] = field(default_factory=list)
    sockets: list[SocketInfo] = field(default_factory=list)


def _parse_vec3(s: str) -> tuple[float, float, float]:
    """解析 "x,y,z" 格式的字符串。"""
    parts = s.strip().split(",")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _parse_vec4(s: str) -> tuple[float, float, float, float]:
    """解析 "x,y,z,w" 格式的字符串。"""
    parts = s.strip().split(",")
    return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))


def parse_gim(xml_str: str) -> GimData:
    """解析 .gim XML 文本，返回 GimData。"""
    out = GimData()

    # 顶层属性
    m = re.search(r'VertexCount\s*=\s*"(\d+)"', xml_str)
    if m:
        out.vertex_count = int(m.group(1))
    m = re.search(r'TriangleCount\s*=\s*"(\d+)"', xml_str)
    if m:
        out.triangle_count = int(m.group(1))
    m = re.search(r'TangentEnable\s*=\s*"(\w+)"', xml_str)
    if m:
        out.tangent_enable = m.group(1).lower() == "true"

    # 包围盒
    m = re.search(r'BoundingInfo\s*=\s*"\(([^)]+)\),\(([^)]+)\),([^"]+)"', xml_str)
    if m:
        out.bounding_center = _parse_vec3(m.group(1))
        out.bounding_half = _parse_vec3(m.group(2))
        out.bounding_radius = float(m.group(3).strip())

    # Submesh 列表
    #
    # ⛔【2026-10-07 v3.97 —— 试过把这两个字段改成可选，**已回退**，别再试】
    #
    # 有一类 `.gim`（charselect 1403 个里 **64 个**、解压树 18515 个里 **156 个**）的
    # `<SubN>` 块里**没有逐子网的顶点/面数**，只有 Name/MtlIdx/BoundingCenter/BoundingHalf；
    # 文件级 `<NeoX>` 头里那两个是**整包总数**，不是逐子网的。实测 **`00129_02`（主基准）**
    # 就是这一类，而且解压树给 `00129_02.cmesh` 配的那份同名 `.gim`
    # （= NPK 的 `gim_005.gim`，md5 6bedd5df…）正是它 —— 配对是错的，
    # 正确的那份在 `dyes\1712903\玲珑00129_02.gim`（vc/tc 与网格相符、子网计数与抓帧真值
    # 9484/1012/7747/2397/13100 **完全一致**）。
    #
    # **为什么不能在这里"容错"**：把两个字段改成可选后，这类 `.gim` 会解析出
    # **N 个子网但计数全 0**；而 `viewer.py` 在 `mesh_data.submeshes` 为空时会**回退用
    # `gim_data.submeshes`**（`viewer.py:1786`）⇒ 渲染出 N 个 0 面子网 ⇒ **整屏空白**
    # （判据 E 主体占屏比 0.8778 → **0.0259**）。
    # 改之前它解析出 **0 个子网**、触发单 submesh 兜底，反而能正常显示。
    #
    # ⇒ 要修就得**连面序/材质索引映射一起解**（见 `viewer_app._find_matching_gim`，
    #    已写好但默认关闭，用 `VIEWER_GIM_FIX=1` 打开排查）。
    # 【2026-10-10 第六十七轮】**放宽：`VertexCount`/`TriangleCount` 改为可选。**
    #
    # 原正则是：
    #     <Sub(\d+)\s+ Name\s*=\s*"([^"]+)"\s+ MtlIdx\s*=\s*"(\d+)"\s+
    #     VertexCount\s*=\s*"(\d+)"\s+ TriangleCount\s*=\s*"(\d+)"
    # ⇒ **两个计数字段必需**。而有一类 `.gim`（如 `石矶00116.gim`，21 条 `<SubN>`）
    #   里 `VertexCount`/`TriangleCount` **整份文件只出现 1 次**（是文件级），
    #   子网级一个都没有 ⇒ 正则一条都不匹配 ⇒ `submeshes` 为空。
    #
    # 之前不改是对的：那时**没有任何地方**能提供逐子网计数，改宽会得到
    #   "N 个子网但计数全 0"，被 `viewer_app` 用来覆盖 `mesh.submeshes` ⇒
    #   渲染 N 个 0 面子网 ⇒ 整屏空白（判据 E 0.8778 → 0.0259）。
    #
    # 现在安全前提已具备（见 `mesh.py::_parse_v4_payload` 第六十七轮注释）：
    #   `.cmesh` v4 自带子网格表，已落到 `mesh.submeshes`（计数精确相符）；
    #   且 `viewer_app` 里那条 `_gim_ok` 判据**仍然拦着**"计数和与 face_count 不符"的情况。
    # ⇒ 放宽后最坏情况只是"多拿到名字/MtlIdx、没有计数"，
    #   而**不会**再出现 0 面子网覆盖（覆盖仍受 `_gim_ok` 把关）。
    submesh_pattern = re.compile(
        r'<Sub(\d+)\s+'
        r'Name\s*=\s*"([^"]+)"\s+'
        r'MtlIdx\s*=\s*"(\d+)"\s+'
        r'(?:VertexCount\s*=\s*"(\d+)"\s*)?'
        r'(?:TriangleCount\s*=\s*"(\d+)"\s*)?'
        r'(?:\s*BoundingCenter\s*=\s*"([^"]*)")?'
        r'(?:\s*BoundingHalf\s*=\s*"([^"]*)")?'
    )
    for m in submesh_pattern.finditer(xml_str):
        idx = int(m.group(1))
        name = m.group(2)
        mtl_idx = int(m.group(3))
        vc = int(m.group(4)) if m.group(4) else 0
        tc = int(m.group(5)) if m.group(5) else 0
        bc = _parse_vec3(m.group(6)) if m.group(6) else (0.0, 0.0, 0.0)
        bh = _parse_vec3(m.group(7)) if m.group(7) else (0.0, 0.0, 0.0)
        out.submeshes.append(SubmeshInfo(
            index=idx, name=name, mtl_idx=mtl_idx,
            vertex_count=vc, triangle_count=tc,
            bounding_center=bc, bounding_half=bh,
        ))

    # 计算 vertex_start / face_start
    v_start = 0
    f_start = 0
    for sm in out.submeshes:
        sm.vertex_start = v_start
        sm.face_start = f_start
        v_start += sm.vertex_count
        f_start += sm.triangle_count

    # GisFiles
    gis_pattern = re.compile(r'<FileName\s+value\s*=\s*"([^"]+)"')
    for m in gis_pattern.finditer(xml_str):
        out.gis_files.append(m.group(1))

    # Sockets（简化解析，先提取名称和绑定类型）
    socket_pattern = re.compile(
        r'<Socket_\d+\s+Name\s*=\s*"([^"]+)"'
        r'(?:\s+BindType\s*=\s*"(\d+)")?'
    )
    for m in socket_pattern.finditer(xml_str):
        name = m.group(1)
        bind_type = int(m.group(2)) if m.group(2) else 0
        out.sockets.append(SocketInfo(
            name=name, bind_type=bind_type,
            position=(0, 0, 0), rotation=(0, 0, 0, 1),
        ))

    return out


def parse_gim_file(path: str | bytes) -> GimData:
    """从文件路径或原始字节解析 .gim。"""
    if isinstance(path, bytes):
        return parse_gim(path.decode("utf-8", errors="replace"))
    with open(path, "r", encoding="utf-8") as f:
        return parse_gim(f.read())

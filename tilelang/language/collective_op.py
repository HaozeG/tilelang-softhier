"""Collective operations exposed on the TileLang language surface.

These ops emit tir.call_intrin nodes that are consumed by downstream compilers
(e.g. Deeploy).  TileLang's own CUDA backend treats them as no-ops — it emits
a stub comment and generates no device code.

Argument encoding conventions (mirrors Deeploy's TilelangVisitor):
    Buffer / region  → tl.region PrimExpr via to_buffer_region()
    str              → tir.StringImm  (empty string encodes "not provided")
    int              → tir.IntImm("int32", v)  (-1 encodes "not provided")
"""

from __future__ import annotations

from tvm import tir
from tilelang.language.frame import has_let_value, get_let_value
from tilelang.utils.language import to_buffer_region, get_buffer_region_from_load


def _get_extents(buf):
    """Extract per-dimension extents from a buffer-like object."""
    if isinstance(buf, tir.Buffer):
        return list(buf.shape)
    if isinstance(buf, tir.BufferRegion):
        return [r.extent for r in buf.region]
    if isinstance(buf, tir.BufferLoad):
        region = get_buffer_region_from_load(buf)
        if region is not None:
            return [r.extent for r in region.region]
        return [tir.IntImm("int32", 1) for _ in buf.indices]
    return []


def _to_region(buf, access_type: str = "rw") -> tir.PrimExpr:
    """Convert a buffer-like object to a tl.region PrimExpr."""
    if isinstance(buf, tir.Var) and has_let_value(buf):
        buf = get_let_value(buf)
    extents = _get_extents(buf)
    return to_buffer_region(buf, access_type=access_type, extents=extents)


def allreduce(buf, reduce_op: str = "sum", axis: str | None = None,
              group: str | None = None) -> tir.PrimExpr:
    """In-place allreduce across the cluster group named `group`.

    Parameters
    ----------
    buf :
        Buffer to reduce in-place.
    reduce_op : str
        Reduction operator: "sum", "max", or "min".
    axis : str or None
        Loop variable name that was group-partitioned (e.g. "bk").
        Passed as a semantic annotation for the downstream compiler;
        does NOT affect the TIR structure.
    group : str or None
        Name of the cluster group.  Falls back to the enclosing
        T.attr("anno", "cluster_group", ...) annotation if None.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.allreduce"),
        _to_region(buf, access_type="rw"),
        tir.StringImm(reduce_op),
        tir.StringImm(axis or ""),
        tir.StringImm(group or ""),
    )


def group_shift(buf, along: str, by: int = 1, group: str | None = None) -> tir.PrimExpr:
    """Ring-shift `buf` across group members along the `along` axis.

    Every rank k sends its tile to rank (k + by) % N along that axis.
    Wrap is always enabled in v1 (Cannon / Systolic rotation pattern).

    Parameters
    ----------
    buf :
        Buffer to shift in-place (source and destination aliased).
    along : str
        Group axis name along which to rotate (must match one of
        ``ClusterGroup.axis_names``).
    by : int
        Step count (positive = forward along the axis).
    group : str or None
        Name of the cluster group.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.group_shift"),
        _to_region(buf, access_type="rw"),
        tir.StringImm(along),
        tir.IntImm("int32", int(by)),
        tir.StringImm(group or ""),
    )


def group_bcast_axis(buf, along: str, from_coord: int = 0,
                     group: str | None = None) -> tir.PrimExpr:
    """Broadcast `buf` along one axis of the group from a given source rank.

    Parameters
    ----------
    buf :
        Buffer to broadcast in-place.
    along : str
        Group axis name along which to broadcast.
    from_coord : int
        Source rank along that axis (``0`` = first rank along the axis).
    group : str or None
        Name of the cluster group.
    """
    # Accept either a static int or a TIR PrimExpr (e.g. a loop Var for
    # dynamic per-rank source selection in 2D SUMMA).
    _fc = from_coord if hasattr(from_coord, "dtype") else tir.IntImm("int32", int(from_coord))
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.group_bcast_axis"),
        _to_region(buf, access_type="rw"),
        tir.StringImm(along),
        _fc,
        tir.StringImm(group or ""),
    )


def broadcast(buf, root: int | None = None, group: str | None = None) -> tir.PrimExpr:
    """Broadcast `buf` from root cluster in `group` to all members.

    Parameters
    ----------
    buf :
        Buffer to broadcast in-place.
    root : int or None
        Root cluster index within the group.  -1 encodes "not provided"
        (Deeploy defaults to the group's root_instance).
    group : str or None
        Name of the cluster group.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.broadcast"),
        _to_region(buf, access_type="rw"),
        tir.IntImm("int32", root if root is not None else -1),
        tir.StringImm(group or ""),
    )


def scatter(src_buf, dst_buf, group: str | None = None) -> tir.PrimExpr:
    """Root distributes src_buf chunks to all group members' dst_buf.

    Parameters
    ----------
    src_buf :
        Root's source buffer.  Size = N_members * chunk_size (root-only).
        On non-root clusters this argument is ignored.
    dst_buf :
        Each member's receive buffer.  Size = chunk_size.
    group : str or None
        Name of the cluster group.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.scatter"),
        _to_region(src_buf, access_type="r"),
        _to_region(dst_buf, access_type="w"),
        tir.StringImm(group or ""),
    )


def gather(src_buf, dst_buf, group: str | None = None) -> tir.PrimExpr:
    """Collect all group members' src_buf chunks into root's dst_buf.

    Parameters
    ----------
    src_buf :
        Each member's source buffer.  Size = chunk_size.
    dst_buf :
        Root's destination buffer.  Size = N_members * chunk_size (root-only).
    group : str or None
        Name of the cluster group.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.gather"),
        _to_region(src_buf, access_type="r"),
        _to_region(dst_buf, access_type="w"),
        tir.StringImm(group or ""),
    )


def alloc_gather_dst(shape, dtype, group: str) -> tir.PrimExpr:
    """Allocate a root-only gather destination buffer (size = N_members * chunk_size).

    This MUST remain as a tir.call_intrin statement node — do NOT convert to
    alloc_buffers.  It appears inside the T.attr("cluster_group", ...) block
    body so that downstream visitors process it with the correct group context.

    Parameters
    ----------
    shape : sequence of ints
        Buffer shape dimensions.
    dtype : str
        Element type string, e.g. "float16".
    group : str
        Name of the cluster group.
    """
    shape_imms = [tir.IntImm("int32", s) for s in shape]
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.alloc_gather_dst"),
        *shape_imms,
        tir.StringImm(str(dtype)),
        tir.StringImm(group),
    )


def alloc_scatter_src(shape, dtype, group: str) -> tir.PrimExpr:
    """Allocate a root-only scatter source buffer (size = N_members * chunk_size).

    Same statement-level semantics as alloc_gather_dst — must NOT be hoisted.

    Parameters
    ----------
    shape : sequence of ints
        Buffer shape dimensions.
    dtype : str
        Element type string, e.g. "float16".
    group : str
        Name of the cluster group.
    """
    shape_imms = [tir.IntImm("int32", s) for s in shape]
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.alloc_scatter_src"),
        *shape_imms,
        tir.StringImm(str(dtype)),
        tir.StringImm(group),
    )


def synchronize(group: str | None = None) -> tir.PrimExpr:
    """Explicit group barrier.  Falls back to global barrier when group is None.

    Parameters
    ----------
    group : str or None
        Name of the cluster group.  Empty string → global barrier.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.synchronize"),
        tir.StringImm(group or ""),
    )


def arrive_wait(sync_id: int, count: int) -> tir.PrimExpr:
    """Split barrier for pipeline-parallel stage boundaries.

    Producer calls arrive; consumer calls wait with matching sync_id.

    Parameters
    ----------
    sync_id : int
        Barrier identifier shared between producer and consumer.
    count : int
        Number of arrivals expected before the barrier releases.
    """
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.arrive_wait"),
        tir.IntImm("int32", sync_id),
        tir.IntImm("int32", count),
    )

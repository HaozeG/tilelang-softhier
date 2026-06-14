# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""Cluster-group and layout annotation helpers.

These helpers emit AttrStmt nodes with stringly-serialized specs that
downstream compilers (Deeploy's TilelangVisitor) parse into
ClusterGroup / TensorLayout IR objects.  TileLang's own CUDA backend
treats them as no-ops.

Spec string grammar (owned by this file — single source of truth):

  cluster_group:
    "<gid>;x=<int>;y=<int>;num_groups=<int>;axes=<a>,<b>[;root=rx,ry]
     [;meta_axes=<m0>[,<m1>]][;meta_shape=<s0>[,<s1>]]"

  layout:
    "<buffer_name>=<clause>[;<clause>...]"
    clause := axis<N>:<axis_name>  |  partial:<op>@<axis_name>

Examples::

    T.cluster_group("tp", x=2, y=1, num_groups=1, axes=("tp", "_"))
    # → T.attr("anno", "cluster_group", "tp;x=2;y=1;num_groups=1;axes=tp,_")

    # split-K with 4 k-instances:
    with T.cluster_group("summa", x=2, y=2, num_groups=4, axes=("x","y"),
                         meta_axes=("k",), meta_shape=(4,)) as (gid, gid_x, gid_y, gid_k):
        ...
    # → spec "summa;x=2;y=2;num_groups=4;axes=x,y;meta_axes=k;meta_shape=4"

    T.tile_layout(C_local, partial=("sum", "tp"))
    # → T.attr("anno", "layout", "C_local=partial:sum@tp")

    T.tile_layout(A_local, axis_map={1: "tp"})
    # → T.attr("anno", "layout", "A_local=axis1:tp")
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence, Tuple

from tvm import tir
from tvm.script.ir_builder import tir as ib_tir
from tvm.script.ir_builder.tir.frame import AttrFrame

__all__ = ["cluster_group", "tile_layout"]


def _make_cluster_group_frame(
    group_id: str,
    spec: str,
    split_axes: Tuple[str, ...] = (),
) -> AttrFrame:
    """Return an AttrFrame subclass that yields group-rank vars from __enter__.

    TileLang's eager builder dispatches ``with`` statements via ``ctx_with``,
    which routes any ``IRBuilderFrame`` through ``with_frame``.  That path
    correctly handles LIFO frame cleanup — in particular, ``T.alloc_fragment``
    calls inside the ``with T.cluster_group():`` body push ``AllocBufferFrame``
    objects on top of the ``AttrFrame``; ``with_frame``'s cleanup pops them in
    reverse order before closing the ``AttrFrame``, so the ``AttrStmt`` body
    captures all statements that appear inside the ``with`` block.

    Returns a tuple ``(gid, gid_x, gid_y, *split_ids)`` where each split_id is
    a ``tir.Var`` named ``_group_id_<axis>_<group_id>``.
    """
    vars_tuple = (
        tir.Var(f"_group_id_{group_id}", "int32"),
        tir.Var(f"_group_id_x_{group_id}", "int32"),
        tir.Var(f"_group_id_y_{group_id}", "int32"),
        *(tir.Var(f"_group_id_{axis}_{group_id}", "int32") for axis in split_axes),
    )

    # Build the underlying AttrFrame first.
    inner: AttrFrame = ib_tir.attr(None, "cluster_group", spec)

    # Subclass that overrides __enter__ to return the vars tuple.
    class _CGFrame(AttrFrame):
        def __enter__(self):
            super().__enter__()
            return vars_tuple

    # Rebind the Python class of the existing C++ object so that
    # isinstance(frame, IRBuilderFrame) is True and __enter__ returns vars.
    inner.__class__ = _CGFrame
    return inner


def cluster_group(
    group_id: str,
    x: int = 1,
    y: int = 1,
    num_groups: int = 1,
    axes: Tuple[str, str] = ("x", "y"),
    root: Optional[Tuple[int, int]] = None,
    split_axes: Sequence[str] = (),
    split_shape: Sequence[int] = (),
    meta_axes: Optional[Sequence[str]] = None,
    meta_shape: Optional[Sequence[int]] = None,
) -> AttrFrame:
    """Declare a 2-D cluster group over the current scope.

    Emits ``T.attr(None, "cluster_group", "<serialized spec>")``
    which Deeploy's TilelangVisitor parses back into a ``ClusterGroup``
    object.  TileLang's own CUDA backend ignores this annotation.

    Parameters
    ----------
    group_id : str
        Unique name for this cluster group.
    x : int
        Group width — ``grid_x_dim`` in ``GridSyncGroupInfo``.
    y : int
        Group height — ``grid_y_dim`` in ``GridSyncGroupInfo``.
    num_groups : int
        Number of group instances tiled across the physical cluster grid.
    axes : (str, str)
        Names for the two intra-group axes, e.g. ``("tp", "dp")``.
    root : (int, int) or None
        (rx, ry) root rank within one group instance.  Defaults to (0, 0).
    split_axes : sequence of str
        Names for axes along which the computation (e.g. K-dimension) is
        split across ``num_groups`` instances.
        ``prod(split_shape)`` must equal ``num_groups``.
        The ``with`` statement returns one extra ``tir.Var`` per split-axis,
        named ``_group_id_<axis>_<group_id>``.
    split_shape : sequence of int
        Size of each split-axis.  Must have the same length as ``split_axes``.
    meta_axes : sequence of str (deprecated)
        Legacy alias for ``split_axes``.
    meta_shape : sequence of int (deprecated)
        Legacy alias for ``split_shape``.

    Returns
    -------
    AttrFrame
        A context manager.  ``__enter__`` returns
        ``(gid, gid_x, gid_y, *split_ids)`` as TIR ``Var`` objects.
    """
    if x < 1 or y < 1 or num_groups < 1:
        raise ValueError(f"cluster_group: x={x}, y={y}, num_groups={num_groups} must all be >= 1")
    if len(axes) != 2:
        raise ValueError(f"cluster_group: axes must be a 2-tuple, got {axes!r}")

    # Accept either split_axes (preferred) or meta_axes (legacy).
    if meta_axes is not None:
        split_axes = tuple(meta_axes)
    else:
        split_axes = tuple(split_axes)
    if meta_shape is not None:
        split_shape = tuple(meta_shape)
    else:
        split_shape = tuple(split_shape)

    if len(split_axes) != len(split_shape):
        raise ValueError(
            f"cluster_group: split_axes and split_shape must have the same length, "
            f"got {split_axes!r} vs {split_shape!r}"
        )
    if split_shape:
        split_prod = math.prod(split_shape)
        if split_prod != num_groups:
            raise ValueError(
                f"cluster_group: prod(split_shape)={split_prod} must equal "
                f"num_groups={num_groups}"
            )
        for name in split_axes:
            if name in axes:
                raise ValueError(
                    f"cluster_group: split_axis '{name}' shadows an intra-group "
                    f"axis name in axes={axes!r}"
                )

    parts = [
        group_id,
        f"x={x}",
        f"y={y}",
        f"num_groups={num_groups}",
        f"axes={axes[0]},{axes[1]}",
    ]
    if root is not None:
        parts.append(f"root={root[0]},{root[1]}")
    if split_axes:
        parts.append(f"split_axes={','.join(split_axes)}")
        parts.append(f"split_shape={','.join(str(s) for s in split_shape)}")
    spec = ";".join(parts)
    return _make_cluster_group_frame(group_id, spec, split_axes=split_axes)


def tile_layout(
    buf,
    axis_map: Optional[Mapping[int, str]] = None,
    partial: Optional[Tuple[str, str]] = None,
):
    """Annotate a tile buffer with a sharding layout relative to the enclosing group.

    Emits ``T.attr("anno", "layout", "<buf_name>=<clauses>")``
    which Deeploy's TilelangVisitor parses into a ``TensorLayout`` object.

    Parameters
    ----------
    buf :
        The buffer being annotated.  Accepts a TIR Buffer, Var, BufferLoad,
        or a plain Python string (escape hatch for testing).
    axis_map : dict[int, str] or None
        Maps tensor axis indices to group axis names.  For example,
        ``{1: "tp"}`` means "tensor axis 1 is sharded along the group
        axis named 'tp'".  Axes not mentioned are implicitly replicated.
    partial : (str, str) or None
        ``(reduce_op, axis_name)`` — marks this buffer as holding a
        partial result awaiting reduction along ``axis_name``.  For
        example, ``("sum", "tp")`` means "this buffer will be summed
        across the 'tp' axis via allreduce".

    Examples
    --------
    ::

        T.tile_layout(C_local, partial=("sum", "tp"))
        T.tile_layout(A_local, axis_map={1: "tp"})
        T.tile_layout(B_local, axis_map={0: "tp"})
    """
    name = _buffer_name(buf)
    clauses = []
    if axis_map:
        for ax, group_axis in sorted(axis_map.items()):
            clauses.append(f"axis{int(ax)}:{group_axis}")
    if partial is not None:
        op, group_axis = partial
        clauses.append(f"partial:{op}@{group_axis}")
    if not clauses:
        raise ValueError("tile_layout: at least one of axis_map or partial is required")
    spec = f"{name}=" + ";".join(clauses)
    return ib_tir.attr(None, "layout", spec)


def _buffer_name(buf) -> str:
    """Extract the C-level name from a TIR buffer/region/var argument."""
    if isinstance(buf, str):
        return buf
    # tir.Buffer
    if hasattr(buf, "name") and isinstance(getattr(buf, "name"), str):
        return str(buf.name)
    # tir.Var or similar with name_hint
    if hasattr(buf, "name_hint"):
        return str(buf.name_hint)
    # tir.BufferLoad — buffer is at .buffer.name
    if hasattr(buf, "buffer") and hasattr(buf.buffer, "name"):
        return str(buf.buffer.name)
    # tir.Buffer accessed via .data.name_hint
    if hasattr(buf, "data") and hasattr(buf.data, "name_hint"):
        return str(buf.data.name_hint)
    raise TypeError(
        f"tile_layout: cannot extract name from {type(buf).__name__}. "
        "Pass a tir.Buffer, tir.Var, BufferLoad, or plain str."
    )

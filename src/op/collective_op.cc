/*!
 * \file src/op/collective_op.cc
 *
 * Register collective operators (allreduce, broadcast, scatter, gather, etc.)
 * as opaque TVM ops.
 *
 * Design notes:
 * - No TLOpBuilder is set for any of these ops.  ParseOperator() therefore
 *   returns an empty TileOperator for them, and TileLang's lowering passes
 *   leave the tir.call_intrin nodes in place — unchanged — for downstream
 *   compilers such as Deeploy to consume.
 * - alloc_gather_dst / alloc_scatter_src are registered as statement-level
 *   ops with kOpaque effect so that TVM never hoists them out of their
 *   enclosing T.attr("cluster_group") block.
 * - The built-in CUDA backend emits a stub comment for each op (handled in
 *   codegen_cuda.cc::VisitStmt_(EvaluateNode)).
 */

#include <tvm/ir/op.h>
#include <tvm/tir/op_attr_types.h>

namespace tvm {
namespace tl {

using namespace tir;

TVM_REGISTER_OP("tl.tileop.allreduce")
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.broadcast")
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.group_shift")
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.group_bcast_axis")
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.scatter")
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.gather")
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

// alloc_gather_dst / alloc_scatter_src must NOT be hoisted by TVM allocate
// promotion passes.  Registering as kOpaque statement-level ops ensures they
// remain as Evaluate(call_intrin(...)) nodes inside the cluster_group AttrStmt
// body where the downstream visitor can process them with the correct group
// context.
TVM_REGISTER_OP("tl.tileop.alloc_gather_dst")
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.alloc_scatter_src")
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.synchronize")
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.arrive_wait")
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));

} // namespace tl
} // namespace tvm

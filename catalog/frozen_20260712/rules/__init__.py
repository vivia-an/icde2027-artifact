"""Invariant rule registry.

Importing this package side-loads all rule modules.
"""
from . import base  # re-export decorators

# Side-import all T0 rule files so @rule registration fires
from . import (T0_attention_head_uniformity,
               T0_checkpoint_preserve_rng, T0_clip_grad_bounded,
               T0_dtype_propagation, T0_evaluator_eval_mode,
               T0_grad_norm_finite,
               T0_initial_lr_present, T0_layer_count,
               T0_loss_per_token_dispersion,
               T0_loss_reduction_correct,
               T0_module_grad_output_alive,
               T0_no_nan_inf, T0_norm_output_rms, T0_optim_lr_positive,
               T0_optim_step_counter,
               T0_param_update_applied, T0_softmax_degenerate,
               T0_token_id_range,
               T1_buffer_replica_cksum_equal,
               T1_comm_dtype_matches_training,
               T1_expert_bias_dtype,
               T1_fwd_output_block_uniformity,
               T1_grad_flow_block_uniformity,
               T1_grad_replica_cksum_equal,
               T1_jitter_dtype, T1_layer_count_strict,
               T1_multi_backward_per_step,
               T1_process_group_size,
               T1_replica_cksum_equal, T1_residual_stream,
               T1_router_attribute, T1_sqrt_decay_direction)

from .base import RuleResult, all_rules, rule

__all__ = ["rule", "RuleResult", "all_rules"]

"""T0: PyTorch-core hook installations."""
from .build_snapshot import take_build_snapshot
from .checkpoint_hook import install_checkpoint_hook
from .dataloader_hook import install_dataloader_hook
from .dist_hook import install_dist_hooks
from .functional_hook import install_functional_hooks
from .loss_hook import install_loss_hooks
from .module_hook import install_module_hooks
from .optim_hook import (install_clip_grad_hook, install_lr_scheduler_hook,
                         install_optim_hooks)

__all__ = [
    "install_dist_hooks",
    "install_module_hooks",
    "install_optim_hooks",
    "install_clip_grad_hook",
    "install_lr_scheduler_hook",
    "install_checkpoint_hook",
    "install_dataloader_hook",
    "install_functional_hooks",
    "install_loss_hooks",
    "take_build_snapshot",
]

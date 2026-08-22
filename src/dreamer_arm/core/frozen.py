"""Frozen rollout views: modules that share live parameter storage but block gradients.

Two things need a no-grad view of an otherwise-trainable module: imagination
rollouts (:meth:`~dreamer_arm.core.actor_critic.ActorCritic._imagine`) and
rollout inference (:meth:`~dreamer_arm.core.model.Dreamer.act`, via each world
model's ``encode_for_act``/``observe_step``/``img_step``). Both want the
*current* trained weights without either (a) letting gradients flow back
through them, or (b) paying for a real second copy of the parameters.

:func:`freeze_clone` deep-copies the module structure (so its own buffers,
dropout state etc. are independent) but repoints every parameter's ``.data``
at the *same storage* as the source, with ``requires_grad=False``. It must be
called again after anything that reallocates the source's storage --
``.to(device)``, ``load_state_dict`` -- since the clone's parameters would
otherwise still point at the old (freed or stale) storage.
"""

from __future__ import annotations

import copy

from torch import nn


def freeze_clone[ModuleT: nn.Module](module: ModuleT) -> ModuleT:
    """Deep-copy ``module``, sharing parameter storage but with grad disabled."""
    frozen = copy.deepcopy(module)
    for (n_o, p_o), (n_n, p_n) in zip(module.named_parameters(), frozen.named_parameters(), strict=True):
        assert n_o == n_n
        p_n.data = p_o.data  # share storage
        p_n.requires_grad_(False)
    return frozen


__all__ = ["freeze_clone"]

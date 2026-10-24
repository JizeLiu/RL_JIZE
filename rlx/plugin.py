"""Single entry point that VERL loads in *every* process (driver, Ray actors, rollout servers).

Activate it by exporting, before ``ray start`` / the training command::

    export VERL_USE_EXTERNAL_MODULES=rlx.plugin
    export RLX_PLUGINS=rlx.selfdistill.trainer      # comma separated, optional

VERL's ``verl/__init__.py`` imports every module listed in ``VERL_USE_EXTERNAL_MODULES`` at import
time, so decorators such as ``@register_trainer`` / ``@register_adv_est`` / ``@register_policy_loss``
run in each worker without patching VERL.

With ``RLX_PLUGINS`` unset this module does nothing: plain GRPO / PPO runs load no extra code and keep
bit-identical behaviour and performance. To add your own algorithm, drop a module that uses one of
VERL's ``register_*`` decorators and list it in ``RLX_PLUGINS``.
"""

from __future__ import annotations

import importlib
import logging
import os

logger = logging.getLogger(__name__)


def load_plugins() -> list[str]:
    loaded = []
    for name in filter(None, (n.strip() for n in os.environ.get("RLX_PLUGINS", "").split(","))):
        importlib.import_module(name)
        loaded.append(name)
    if loaded:
        logger.info("rlx plugins loaded: %s", loaded)
    return loaded


LOADED = load_plugins()

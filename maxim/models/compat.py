"""Narrow compatibility shims for the pinned Flax/JAX versions.

Flax 0.11.2 still forwards ``concrete=...`` to ``jax.remat``, an option that
JAX 0.10 removed. ``install_flax_remat_compat`` replaces the ``jax`` reference
*inside ``flax.core.lift`` only* with a proxy whose ``remat``/``checkpoint``
drop ``concrete=False``. Every other attribute is delegated to the real
``jax`` module, and no other module is affected.
"""

import types

import jax


def _strip_concrete(function):
  def wrapper(fun=None, *, concrete=False, **kwargs):
    if concrete:
      raise NotImplementedError("concrete=True is not supported by this JAX.")
    if fun is None:
      return lambda inner: function(inner, **kwargs)
    return function(fun, **kwargs)

  return wrapper


class _JaxRematProxy(types.ModuleType):

  def __getattr__(self, name):
    return getattr(jax, name)


_JaxRematProxy.remat = staticmethod(_strip_concrete(jax.checkpoint))
_JaxRematProxy.checkpoint = staticmethod(_strip_concrete(jax.checkpoint))


def install_flax_remat_compat():
  """Idempotently installs the proxy. Returns True if it is active."""
  import flax.core.lift as lift

  if isinstance(lift.jax, _JaxRematProxy):
    return True
  try:
    jax.checkpoint(lambda value: value, concrete=False)
    return False  # This JAX still accepts the option; nothing to patch.
  except (TypeError, NotImplementedError):
    pass
  lift.jax = _JaxRematProxy("jax_remat_compat")
  return True

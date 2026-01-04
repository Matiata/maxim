import importlib
import collections
import ml_collections
import io
import jax.numpy as jnp
from jax import random
import numpy as np
import tensorflow as tf
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

# !git clone https://github.com/Matiata/maxim.git
# %cd /content/maxim

SEED = 42
NUM_EXPERTS = 5
CKPT_PATH = "./ckpts/maxim_ckpt_Deblurring_GoPro_checkpoint.npz"
TASK = "deblur"
_MODEL_CONFIGS = {
    "variant": "",
    "dropout_rate": 0.0,
    "num_outputs": 3,
    "use_bias": True,
    "num_supervision_scales": 3,
}
_MODEL_VARIANT_DICT = {
    "denoise": "S-3",
    "deblur": "S-3",
    "derain": "S-2",
    "dehaze": "S-2",
    "enhance": "S-2",
}
 
def recover_tree(keys, values):
    """Recovers a tree as a nested dict from flat names and values.

    This function is useful to analyze checkpoints that are saved by our programs
    without need to access the exact source code of the experiment. In particular,
    it can be used to extract an reuse various subtrees of the scheckpoint, e.g.
    subtree of parameters.
    Args:
      keys: a list of keys, where '/' is used as separator between nodes.
      values: a list of leaf values.
    Returns:
      A nested tree-like dict.
    """
    tree = {}
    sub_trees = collections.defaultdict(list)
    for k, v in zip(keys, values):
        if "/" not in k:
            tree[k] = v
        else:
            k_left, k_right = k.split("/", 1)
            sub_trees[k_left].append((k_right, v))
    for k, kv_pairs in sub_trees.items():
        k_subtree, v_subtree = zip(*kv_pairs)
        tree[k] = recover_tree(k_subtree, v_subtree)
    return tree


def get_params(ckpt_path):
    """Get params checkpoint."""

    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        data = f.read()
    values = np.load(io.BytesIO(data))
    params = recover_tree(*zip(*values.items()))
    params = params["opt"]["target"]

    return params


def load_maxim_backbone(params_moe, params_maxim):
    moe = unfreeze(params_moe)
    flat_moe = flatten_dict(moe)
    flat_maxim = flatten_dict(params_maxim)

    for k, v in flat_maxim.items():
        if k in flat_moe and flat_moe[k].shape == v.shape:
            flat_moe[k] = v

    return freeze(unflatten_dict(flat_moe))


maxim_mod = importlib.import_module("maxim.models.maxim")
router_mod = importlib.import_module("maxim.models.router")
moe_mod = importlib.import_module("maxim.models.moe")

maxim_configs = ml_collections.ConfigDict(_MODEL_CONFIGS)
maxim_configs.variant = _MODEL_VARIANT_DICT[TASK]
maxim_model = maxim_mod.Model(**maxim_configs)

router = router_mod.RouterModel()
experts = [
    moe_mod.ExpertHead(),  # denoise
    moe_mod.ExpertHead(),  # deblur
    moe_mod.ExpertHead(),  # derain
    moe_mod.ExpertHead(),  # dehaze
    moe_mod.ExpertHead(),  # enhance
]
moe_model = moe_mod.MaximMoE(maxim_model, router, experts)

x = jnp.zeros((1, 256, 256, 3))
rng = random.PRNGKey(SEED)
params_maxim = get_params(CKPT_PATH)
variables = moe_model.init(rng, x, train=True)
params_moe = variables["params"]
batch_stats = variables.get("batch_stats")

params_moe = load_maxim_backbone(params_moe, params_maxim)
flat_maxim = flatten_dict(params_maxim)
final_key = 'stage_1_output_conv_0'
final_kernel = flat_maxim[(final_key, "kernel")]
final_bias = flat_maxim[(final_key, "bias")]

params_moe = unfreeze(params_moe)
for e in range(NUM_EXPERTS):
    params_moe[f"experts_{e}"]["output_conv"]["kernel"] = final_kernel
    params_moe[f"experts_{e}"]["output_conv"]["bias"] = final_bias

params_moe = freeze(params_moe)

y_maxim = maxim_model.apply({"params": params_maxim}, x, train=False)
y_maxim = y_maxim[-1][-1]
y_moe, gates = moe_model.apply({"params": params_moe, 'batch_stats': batch_stats}, x, train=False)

diff = jnp.mean(jnp.abs(y_maxim - y_moe))
print(diff)
print(jnp.mean(gates, axis=0))
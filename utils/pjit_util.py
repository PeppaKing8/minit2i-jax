import enum
import functools
import jax
import jax.tree_util as jtu
import numpy as np
from jax.sharding import Mesh as M, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils, multihost_utils
from jax.experimental.pjit import pjit
from utils.logging_util import print0, Emoji
from typing import Tuple

_mesh = None
_jitted_rand_func = {}

def ddp_shard():
  """NamedSharding that shards the first (batch) dim across all mesh axes."""
  assert _mesh is not None, "Mesh not initialized. Call prepare_pjit_funcs first."
  return NamedSharding(_mesh, P(tuple(_mesh.axis_names),))

def enforce_ddp(x):
  """Apply DDP sharding constraint to all leaves of a pytree."""
  if _mesh is None:
    # If the pjit mesh hasn't been initialized, silently skip DDP constraints.
    return x

  def _try_ddp(x):
    if not hasattr(x, "shape") or x.ndim == 0:
      return x
    if x.shape[0] % jax.device_count() != 0:
      return x
    return jax.lax.with_sharding_constraint(x, ddp_shard())
  return jax.tree.map(_try_ddp, x)
  
def ddp_rand_func(rand_type="normal", shard="ddp"):
  """Return a jitted random function with DDP sharding on output.

  Args:
    rand_type: "normal" or "uniform".
    shard: "ddp".
  """
  if shard == "ddp":
    if _mesh is None:
      # Fallback for non-mesh contexts (single-host/single-device inference).
      if rand_type == "normal":
        return jax.random.normal
      if rand_type == "uniform":
        return jax.random.uniform
      raise ValueError(f"Unknown rand_type: {rand_type}")
    sharding = ddp_shard()
  else:
    raise ValueError(f"Unknown shard type: {shard}")
  func_key = (rand_type, shard)
  if func_key not in _jitted_rand_func:
    if rand_type == "normal":
      _jitted_rand_func[func_key] = jax.jit(jax.random.normal, out_shardings=sharding, static_argnums=(1, 2))
    elif rand_type == "uniform":
      _jitted_rand_func[func_key] = jax.jit(jax.random.uniform, out_shardings=sharding, static_argnums=(1, 2))
    else:
      raise ValueError(f"Unknown rand_type: {rand_type}")
  return _jitted_rand_func[func_key]

TOPOLOGIES = {
  "v6": {
    16: (4, 4), # v6e-16
    32: (8, 4),
    64: (8, 8),
    128: (8, 16),
  },
  "v5": {
    8: (4, 2), # v5p-16
    16: (4, 4), # v5-32
    32: (2, 4, 4), # v5-64
    64: (4, 4, 4), # v5-128
  },
  "v4": {
    4: (4, 1), # v4-8
    16: (4, 4), # v4-32
    32: (2, 4, 4), # v4-64
    64: (4, 4, 4), # v4-128
  }
}

def _get_shape(p):
  if hasattr(p, 'shape'):
    return "shape", p.shape
  elif hasattr(p, 'value'):
    return "value.shape", p.value.shape
  elif isinstance(p, Tuple):
    return "self", p
  elif isinstance(p, int) or isinstance(p, float):
    return "scalar", ()
  else:
    return "unk", p
  
def _path_keys(path):
  '''
  Convert a path (tuple of keys) to a list of string keys.
  '''
  keys = []
  for k in path:
    if hasattr(k, "key"):
      keys.append(str(k.key))
    elif hasattr(k, "name"):
      keys.append(str(k.name))
    elif hasattr(k, "idx"):
      keys.append(str(k.idx))
    else:
      keys.append(str(k))
  return keys

class MeshMode(enum.Enum):
  DATA = enum.auto()
  MODEL = enum.auto()
  
def validate_sharding_spec(leaf, spec, mesh_shape: dict):
  """
  Check if the leaf shape is divisible by the mesh dimensions specified in spec.
  """
  if spec is None:
    return True, "No spec provided"
  
  msg, shape = _get_shape(leaf)
  if msg == "unk":
    return False, f"Cannot get shape of leaf: {leaf}"
  
  shape_len = len(shape)
  spec_len = len(spec)
  
  if shape_len < spec_len:
    return False, f"Leaf shape {shape} has fewer dimensions than spec {spec}"
  
  for i in range(spec_len):
    axis_rule = spec[i]
    if axis_rule is None:
      continue
        
    dim_size = shape[i]
    divisor = 1
    
    try:
      if isinstance(axis_rule, str):
        divisor = mesh_shape.get(axis_rule, 0)
      elif isinstance(axis_rule, (tuple, list)):
        for ax in axis_rule:
          divisor *= mesh_shape.get(ax, 0)
      else:
        return False, f"Unknown axis rule type: {type(axis_rule)}"
    except Exception as e:
      return False, f"Error parsing axis rule {axis_rule}: {str(e)}"

    if divisor == 0:
      return False, f"Axis rule {axis_rule} not found in mesh {mesh_shape}"

    if dim_size % divisor != 0:
      return False, f"Dim {i} size {dim_size} is not divisible by mesh axes {axis_rule} (total size {divisor})"

  return True, "ok"

def shard_cpu_state_to_mesh(cpu_state, mesh, partition_specs):
    sharding_tree = jax.tree_util.tree_map(
        lambda spec: NamedSharding(mesh, spec),
        partition_specs
    )

    def _callback_fn(leaf, index):
        return leaf[index]

    def _to_global_array(leaf, sharding):
        return jax.make_array_from_callback(
            np.shape(leaf),
            sharding,
            lambda index: _callback_fn(leaf, index)
        )

    return jax.tree_util.tree_map(_to_global_array, cpu_state, sharding_tree)

def apply_spec_to_last_dims(leaf, spec):
  """
  Apply the given PartitionSpec to the last dimensions of the leaf.
  Returns a new PartitionSpec with None prepended as needed.
  """
  msg, shape = _get_shape(leaf)
  if msg == "unk":
    raise ValueError(f"Cannot get shape of leaf: {leaf}")
  
  shape_len = len(shape)
  spec_len = len(spec)
  
  if shape_len < spec_len: # by default, prune spec
    new_spec = list(spec[-shape_len:]) if shape_len > 0 else []
    return P(*new_spec)
  
  new_spec = [None] * (shape_len - spec_len) + list(spec)
  return P(*new_spec)

def get_spec_dict(tree, mesh: M, param_mode: MeshMode, sharding_mode: str):
  length = len(mesh.shape)
  assert length == 2 or length == 3, f"Only 2D or 3D mesh supported, got {length}D: {mesh.shape}"
  
  prepared_spec_dict = None
  
  if sharding_mode == 'ddp':
    if param_mode == MeshMode.MODEL:
      prepared_spec_dict = jax.tree_map(lambda _: P(), tree)
    
    if param_mode == MeshMode.DATA:
      prepared_spec_dict = jax.tree_map(lambda _: P(tuple(mesh.axis_names),), tree)
  
  elif sharding_mode == 'hsdp':
    if param_mode == MeshMode.MODEL:
      model_shard = P(mesh.axis_names[-1],)
      prepared_spec_dict = jax.tree_map(lambda l: apply_spec_to_last_dims(l, model_shard), tree)

    if param_mode == MeshMode.DATA:
      prepared_spec_dict = jax.tree_map(lambda _: P(tuple(mesh.axis_names),), tree)
  
  elif sharding_mode == 'fsdp':
    if param_mode == MeshMode.MODEL:
      model_shard = P(mesh.axis_names[-1], tuple(mesh.axis_names[:-1]))
      prepared_spec_dict = jax.tree_map(lambda l: apply_spec_to_last_dims(l, model_shard), tree)
    
    if param_mode == MeshMode.DATA:
      data_shard = P(tuple(mesh.axis_names),)
      prepared_spec_dict = jax.tree_map(lambda _: data_shard, tree)
    
  if prepared_spec_dict is None:
    raise ValueError(f"Unsupported sharding mode: {sharding_mode} with param mode: {param_mode}")
  
  # now, validate all specs
  mesh_info_dict = {name: size for name, size in zip(mesh.axis_names, mesh.devices.shape)}
  
  tree_flat_with_path, _ = jtu.tree_flatten_with_path(tree)
  spec_flat_with_path, treedef = jtu.tree_flatten_with_path(prepared_spec_dict)
  new_spec_flat_with_path = []
  
  # sanity check
  assert len(tree_flat_with_path) == len(spec_flat_with_path), \
    f"Tree and spec length mismatch: {len(tree_flat_with_path)} vs {len(spec_flat_with_path)}"
    
  for (path, leaf), (_path, spec) in zip(tree_flat_with_path, spec_flat_with_path):
    if path != _path:
      raise ValueError(f"Path mismatch between tree and spec at {path} vs {_path}")
    
    valid, reason = validate_sharding_spec(leaf, spec, mesh_info_dict)
    if not valid:
      print0(f"{Emoji.WARNING} Warning: invalid sharding spec {spec} for leaf at {jtu.keystr(path)} with shape {_get_shape(leaf)}. Reason: {reason}", flush=True)
      new_spec_flat_with_path.append((path, P()))  # fallback to no sharding
    else:
      new_spec_flat_with_path.append((path, spec))
      
  prepared_spec_dict = jtu.tree_unflatten(treedef, [s for _, s in new_spec_flat_with_path])
  
  return prepared_spec_dict
  
def get_mesh() -> M:
  tpu_type = jax.local_devices()[0].device_kind
  GDC = jax.device_count()  # global device count
  mesh_tuple = None
  for topo_name, topo_dict in TOPOLOGIES.items():
    if topo_name not in tpu_type.lower():
      continue
    if GDC not in topo_dict:
      raise ValueError(f"Unsupported Global Device Count {GDC} for TPU type {tpu_type}")
    mesh_tuple = topo_dict[GDC]
    break

  if mesh_tuple is None:
    raise ValueError(f"Unsupported TPU type: {tpu_type}")
  
  # create mesh
  device_mesh = mesh_utils.create_device_mesh(
    mesh_tuple,
    allow_split_physical_axes=(len(mesh_tuple) > 2),
  ) 
  mesh = M(device_mesh, tuple(f"AXIS_{i}" for i in range(len(mesh_tuple))))
  return mesh
    
def prepare_pjit_funcs(mode: str = 'ddp'):
  print0(f"{Emoji.ROCKET} Setting up FSDP mesh and functions for mode='{mode}' ...", flush=True)
  
  global _mesh, _jitted_rand_func
  _jitted_rand_func = {}  # reset cache
  _mesh = mesh = get_mesh()
  mesh_dim_size = int(np.prod(mesh.devices.shape))
  mesh_dict = {name: size for name, size in zip(mesh.axis_names, mesh.devices.shape)}
  
  print0(f"{Emoji.INFO} Mesh: {mesh.devices.shape}", flush=True)
  print0(f"{Emoji.INFO} Total Mesh Size for FSDP: {mesh_dim_size}", flush=True)
  print0(f"{Emoji.INFO} Mesh Dict: {mesh_dict}", flush=True)
  
  # get_partition_spec: input a pytree, return its PartitionSpec over batch
  def get_partition_spec(tree, param_mode: MeshMode):
    return get_spec_dict(tree, mesh, param_mode=param_mode, sharding_mode=mode)
  
  # pjit_all_gather: input a [LOCAL] pytree, return a [GLOBAL] version of it by all-gathering over batch
  def pjit_all_gather(tree):
    return multihost_utils.host_local_array_to_global_array(tree, mesh, P(tuple(mesh.axis_names),))
  
  # pjit_reduce_scatter: input a [GLOBAL] pytree, return a [LOCAL] version of it
  def pjit_reduce_scatter(tree, param_mode=MeshMode.DATA):
    spec = get_partition_spec(tree, param_mode=param_mode)
    return multihost_utils.global_array_to_host_local_array(tree, mesh, spec)
  
  # pjit_compile: a wrapper of pjit, but with mesh context
  def pjit_compile(fn, in_shardings, out_shardings):
    jitted = pjit(
      fn,
      in_shardings=in_shardings,
      out_shardings=out_shardings,
    )
    
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
      with M(mesh.devices, mesh.axis_names):
        return jitted(*args, **kwargs)
    return wrapped
      
  return mesh, get_partition_spec, pjit_all_gather, pjit_reduce_scatter, pjit_compile


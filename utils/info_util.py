import utils.state_util as state_util
from utils.logging_util import print0


# Function to print number of parameters
def print_params(params):
    params_flatten = state_util.flatten_state_dict(params)

    total_params = 0
    max_length = max(len(k) for k in params_flatten)
    
    def _get_shape(p):
        if hasattr(p, 'shape'):
            return p.shape
        elif hasattr(p, 'value'):
            return p.value.shape
        else:
            return "unknown"
        
    def _get_size(p):
        if hasattr(p, 'size'):
            return p.size
        elif hasattr(p, 'value'):
            return p.value.size
        else:
            return "unknown"
    
    max_shape = max(len(f"{_get_shape(p)}") for p in params_flatten.values())
    max_digits = max(len(f"{_get_size(p):,}") for p in params_flatten.values())
    print0('-' * (max_length + max_digits + max_shape + 8), flush=True)
    for name, param in params_flatten.items():
        layer_params = _get_size(param)
        str_layer_shape = f"{_get_shape(param)}".rjust(max_shape) 
        str_layer_params = f"{layer_params:,}".rjust(max_digits)
        print0(f" {name.ljust(max_length)} | {str_layer_shape} | {str_layer_params} ", flush=True)
        total_params += layer_params
    print0('-' * (max_length + max_digits + max_shape + 8), flush=True)
    print0(f"Total parameters: {total_params:,}", flush=True)

def print_param_shapes(param_shapes):
    params_flatten = state_util.flatten_state_dict(param_shapes)

    max_length = max(len(k) for k in params_flatten)
    max_shape = max(len(f"{p}") for p in params_flatten.values())
    print0('-' * (max_length + max_shape + 5), flush=True)
    for name, shape in params_flatten.items():
        str_layer_shape = f"{shape}".rjust(max_shape) 
        print0(f" {name.ljust(max_length)} | {str_layer_shape} ", flush=True)
    print0('-' * (max_length + max_shape + 5), flush=True)
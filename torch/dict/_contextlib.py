import contextlib

# This is a copy from https://github.com/pytorch/pytorch/blob/main/torch/utils/_contextlib.py#L120
# We use it for compatibility with torch >= 1.10 where the implementation fails
# for some tests in torchrl.

# Extra utilities for working with context managers that should have been
# in the standard library but are not

import functools
import inspect
import sys
import warnings
from typing import Any, Callable, cast, TypeVar

import numpy as np

from torch.compiler import is_compiling

# TD cm functions
LAST_OP_MAPS = {}


def _reverse_lock(self, args, kwargs, out):
    return self.unlock_()


LAST_OP_MAPS["lock_"] = _reverse_lock


def _reverse_unlock(self, args, kwargs, out):
    return self.lock_()


LAST_OP_MAPS["unlock_"] = _reverse_unlock


def _reverse_transpose(self, args, kwargs, out):
    dim0, dim1 = args
    if not out.is_locked:
        return out.update(self.transpose(dim0, dim1), inplace=False)
    else:
        return out.update_(self.transpose(dim0, dim1))


LAST_OP_MAPS["transpose"] = _reverse_transpose


def _reverse_flatten_keys(self, args, kwargs, out):
    sep = args[0] if args else "."
    if not out.is_locked:
        return out.update(self.unflatten_keys(sep), inplace=False)
    else:
        return out.update_(self.unflatten_keys(sep))


LAST_OP_MAPS["flatten_keys"] = _reverse_flatten_keys


def _reverse_unflatten_keys(self, args, kwargs, out):
    sep = args[0] if args else "."
    if not out.is_locked:
        return out.update(self.flatten_keys(sep), inplace=False)
    else:
        return out.update_(self.flatten_keys(sep))


LAST_OP_MAPS["unflatten_keys"] = _reverse_unflatten_keys


def _reverse_flatten(self, args, kwargs, out):
    if len(args) == 2:
        dim0, dim1 = args
    elif len(args) == 1:
        dim0 = args[0]
        dim1 = kwargs.get("end_dim", -1)
    else:
        dim0 = kwargs.get("start_dim", 0)
        dim1 = kwargs.get("end_dim", -1)
    if dim1 < 0:
        dim1 = out.ndim + dim1
    if dim0 < 0:
        dim0 = out.ndim + dim0

    if not out.is_locked:
        return out.update(
            self.unflatten(dim0, out.shape[dim0 : dim1 + 1]), inplace=False
        )
    else:
        return out.update_(self.unflatten(dim0, out.shape[dim0 : dim1 + 1]))


LAST_OP_MAPS["flatten"] = _reverse_flatten


def _reverse_unflatten(self, args, kwargs, out):
    if args:
        dim0 = args[0]
        if len(args) > 1:
            unflattened_size = args[1]
        else:
            unflattened_size = kwargs.get("unflattened_size")
    else:
        dim0 = kwargs.get("dim")
        unflattened_size = kwargs.get("unflattened_size")
    if dim0 < 0:
        dim0 = out.ndim + dim0
    dim1 = dim0 + len(unflattened_size) - 1
    if not out.is_locked:
        unflattened = self.flatten(dim0, dim1)
        return out.update(unflattened, inplace=False)
    else:
        unflattened = self.flatten(dim0, dim1)
        return out.update_(unflattened)


LAST_OP_MAPS["unflatten"] = _reverse_unflatten


def _reverse_permute(self, args, kwargs, out):
    from tensordict.utils import _get_shape_from_args

    dims_list = _get_shape_from_args(*args, kwarg_name="dims", **kwargs)
    dims_list = [dim if dim >= 0 else self.ndim + dim for dim in dims_list]
    # inverse map
    inv_dims_list = np.argsort(dims_list)
    if not out.is_locked:
        return out.update(self.permute(inv_dims_list), inplace=False)
    else:
        return out.update_(self.permute(inv_dims_list))


LAST_OP_MAPS["permute"] = _reverse_permute


def _reverse_view(self, args, kwargs, out):
    if not out.is_locked:
        return out.update(self.view(out.shape), inplace=False)
    else:
        return out.update_(self.view(out.shape))


LAST_OP_MAPS["view"] = _reverse_view


def _reverse_unsqueeze(self, args, kwargs, out):
    if args:
        (dim,) = args
    elif kwargs:
        dim = kwargs["dim"]
    else:
        raise RuntimeError(
            "Cannot use td.unsqueeze() as a decorator if the dimension is implicit."
        )
    if not out.is_locked:
        return out.update(self.squeeze(dim), inplace=False)
    else:
        return out.update_(self.squeeze(dim))


LAST_OP_MAPS["unsqueeze"] = _reverse_unsqueeze


def _reverse_squeeze(self, args, kwargs, out):
    if args:
        (dim,) = args
    elif kwargs:
        dim = kwargs["dim"]
    else:
        raise RuntimeError(
            "Cannot use td.squeeze() as a decorator if the dimension is implicit."
        )
    if not out.is_locked:
        return out.update(self.unsqueeze(dim), inplace=False)
    else:
        return out.update_(self.unsqueeze(dim))


LAST_OP_MAPS["squeeze"] = _reverse_squeeze


def _reverse_to_module(self, args, kwargs, out):
    try:
        with (
            out.unlock_()
            if not is_compiling() and out is not None
            else contextlib.nullcontext()
        ):
            return self.to_module(*args, **kwargs, swap_dest=out)
    except AttributeError:
        # This is a bit unsafe but we assume that out won't have an unlock_() if it's not a TD
        raise RuntimeError(
            "to_module cannot be used as a decorator when return_swap=False."
        )


LAST_OP_MAPS["to_module"] = _reverse_to_module

# -*- coding: utf-8 -*-

from torch.testing._internal.common_utils import run_tests

# === Sparsity ===

# Kernels
from ao.test_sparse_kernels import TestQuantizedSparseKernels  # noqa: F401
from ao.test_sparse_kernels import TestQuantizedSparseLayers  # noqa: F401

if __name__ == '__main__':
    run_tests()

from __future__ import annotations

from typing import Tuple, type_check_only

NestedKey = type_check_only(str | Tuple["NestedKeyType", ...])

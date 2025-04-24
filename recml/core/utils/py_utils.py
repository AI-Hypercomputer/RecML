# Copyright 2024 RecML authors <recommendations-ml@google.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Miscellaneous utilities."""

from collections.abc import Callable
import inspect
from typing import Any


def has_argument(fn: Callable[..., Any], arg_name: str) -> bool:
  """Checks if a function has an argument with a given name."""
  params = inspect.signature(fn).parameters.values()
  param_names = [v.name for v in params]
  has_arg = arg_name in param_names
  has_kw_args = any([v.kind == inspect.Parameter.VAR_KEYWORD for v in params])
  return has_arg or has_kw_args

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
"""Fiddle configuration utilities."""

import inspect
from typing import Any

from absl import flags
from fiddle import absl_flags as fdl_flags
from fiddle.absl_flags import utils as fdl_flags_utils
from fiddle.experimental import auto_config


class FiddleFlag(fdl_flags.FiddleFlag):
  """ABSL flag class for a fiddle configuration.

  Wraps the fiddle flag to allow for using local module fiddlers in the module
  where the base config is defined.

  See the documentation of the parent class for more details.
  """

  def _initial_config(self, expression: str):
    call_expr = fdl_flags_utils.CallExpression.parse(expression)
    base_name = call_expr.func_name
    base_fn = fdl_flags_utils.resolve_function_reference(
        base_name,
        fdl_flags_utils.ImportDottedNameDebugContext.BASE_CONFIG,
        self.default_module,
        self.allow_imports,
        "Could not init a buildable from",
    )
    self.default_module = inspect.getmodule(base_fn)
    try:
      if auto_config.is_auto_config(base_fn):
        cfg = base_fn.as_buildable(*call_expr.args, **call_expr.kwargs)
      else:
        cfg = base_fn(*call_expr.args, **call_expr.kwargs)
    except (AttributeError, ValueError) as e:
      raise ValueError(
          f"Failed to init a buildable from expression: {expression}."
      ) from e

    if cfg is None:
      raise ValueError(
          f"Could not init a buildable from {expression}. Make sure the"
          " function name is valid and that it returns a fiddle buildable."
      )
    return cfg

  def _parse_config(self, command: str, expression: str) -> None:
    if self._initial_config_expression:
      raise ValueError(
          "Only one base configuration is permitted. Received "
          f"{command}:{expression} after "
          f"{self.first_command}:{self._initial_config_expression} was"
          " already provided."
      )
    else:
      self._initial_config_expression = expression

    if command == "config":
      self.value = self._initial_config(expression)
    else:
      raise ValueError(
          f"Unsupported config command: {command}. Supported commands are:"
          " ['config', 'fiddler', 'set']."
      )


def DEFINE_fiddle_config(  # pylint: disable=invalid-name
    name: str,
    *,
    default: Any = None,
    help_string: str,
    pyref_policy: Any | None = None,
    flag_values: flags.FlagValues = flags.FLAGS,
    required: bool = False,
) -> flags.FlagHolder[Any]:
  r"""Declare and define an fiddle config line flag object.

  When used in a python binary, after the flags have been parsed from the
  command line, this command line flag object contain a `fdl.Config` of the
  object.

  Example usage in a python binary:
  ```
  _EXPERIMENT_CONFIG = DEFINE_experiment_config(
      "experiment_cfg", help_string="My experiment config",
  )

  def experiment() -> fdl.Config[Experiment]:
    return fdl.Config(Experiment,...)

  def set_steps(experiment_cfg: fdl.Config[Experiment], steps: int):
    experiment_cfg.task.trainer.train_steps = steps

  def main(argv):
    experiment_cfg = _EXPERIMENT_CONFIG.value
    experiment = fdl.build(experiment_cfg)
    run_experiment(experiment, mode="train_and_eval")

  if __name__ == "__main__":
    app.run(main)
  ```

  results in the `_EXPERIMENT_CONFIG.value` set to a fiddle configuration of the
  experiment with all the command line flags applied in the order they were
  passed in.

  Args:
    name: name of the command line flag.
    default: default value of the flag.
    help_string: help string describing what the flag does.
    pyref_policy: a policy for importing references to Python objects.
    flag_values: the ``FlagValues`` instance with which the flag will be
      registered. This should almost never need to be overridden.
    required: bool, is this a required flag. This must be used as a keyword
      argument.

  Returns:
    A handle to defined flag.
  """
  return flags.DEFINE_flag(
      FiddleFlag(
          name=name,
          default_module=None,
          default=default,
          pyref_policy=pyref_policy,
          parser=flags.ArgumentParser(),
          serializer=fdl_flags.FiddleFlagSerializer(pyref_policy=pyref_policy),
          help_string=help_string,
      ),
      flag_values=flag_values,
      required=required,
  )

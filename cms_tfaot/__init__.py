# coding: utf-8

"""
Lightweight python package providing tools for AOT model deployment.
"""

from __future__ import annotations

__all__ = [
    "HeaderData",
    "common_header_data",
    "detect_models",
    "load_and_normalize_config",
    "compile_model",
    "parse_header",
    "create_wrapper",
]

import os
import re
import yaml
import shutil
import tempfile
import collections

from typing import Any


this_dir = os.path.dirname(os.path.abspath(__file__))


# structure describing content of an aot compiled header file
HeaderData = collections.namedtuple("HeaderData", [
    "batch_size",
    "prefix",
    "namespace",
    "class_name",
    "n_args",
    "arg_counts",
    "arg_counts_no_batch",
    "n_res",
    "res_counts",
    "res_counts_no_batch",
])

# list of entries that are common to all header files, independent of batch sizes
common_header_data = [
    "prefix",
    "namespace",
    "class_name",
    "n_args",
    "n_res",
    "arg_counts_no_batch",
    "res_counts_no_batch",
]


def load_and_normalize_config(config_file: str) -> dict[str, Any]:
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # perform checks and fill in defaults
    if "model" not in config:
        raise ValueError(f"missing 'model' entry in {config_file}")
    if not isinstance(config["model"], dict):
        raise ValueError(f"misconfigured 'model' entry in {config_file}")

    if "name" not in config["model"]:
        raise ValueError(f"missing 'model.name' entry in {config_file}")

    if "version" not in config["model"]:
        raise ValueError(f"missing 'model.version' entry in {config_file}")

    if not config["model"].get("serving_key"):
        config["serving_key"] = "serving_default"

    if not config["model"].get("saved_model"):
        config["model"]["saved_model"] = "saved_model"
    config_dir = os.path.dirname(config_file)
    config["model"]["saved_model"] = os.path.normpath(os.path.join(config_dir, config["model"]["saved_model"]))
    if not os.path.exists(config["model"]["saved_model"]):
        raise ValueError(f"'model.saved_model' directory {config['model']['saved_model']} does not exist")

    if "compilation" not in config:
        raise ValueError(f"missing 'compilation' entry in {config_file}")
    if not isinstance(config["compilation"], dict):
        raise ValueError(f"misconfigured 'compilation' entry in {config_file}")

    if "batch_sizes" not in config["compilation"]:
        raise ValueError(f"missing 'compilation.batch_sizes' entry in {config_file}")
    elif not isinstance(config["compilation"]["batch_sizes"], list):
        raise ValueError(f"misconfigured 'compilation.batch_sizes' entry in {config_file}")

    if not config["compilation"].get("namespace"):
        config["compilation"]["namespace"] = "cms_tfaot"

    if not config["compilation"].get("class_name"):
        config["compilation"]["class_name"] = config["model"]["name"] + r"_bs{}"
    elif r"{}" not in config["compilation"].get("class_name"):
        raise ValueError(rf"misconfigured 'compilation.class_name' entry in {config_file} (missing {{}})")

    return config


def compile_model(config: dict[str, Any], output_dir: str) -> tuple[list[str], list[str]]:
    from cmsml.scripts.compile_tf_graph import compile_tf_graph

    with tempfile.TemporaryDirectory() as tmp_dir:
        compile_class = config["compilation"]["class_name"]
        if config["compilation"]["namespace"]:
            compile_class = f"{config['compilation']['namespace']}::{compile_class}"

        compile_tf_graph(
            model_path=config["model"]["saved_model"],
            output_path=tmp_dir,
            batch_sizes=config["compilation"]["batch_sizes"],
            input_serving_key=config["model"]["serving_key"],
            compile_prefix=config["model"]["name"] + r"_bs{}",
            compile_class=compile_class,
        )
        aot_dir = os.path.join(tmp_dir, "aot")

        # prepare output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # move only certain files
        header_files = []
        object_files = []
        for path in os.listdir(aot_dir):
            if re.match(r"^.*_bs\d+\.h$", path):
                header_files.append(path)
            elif re.match(r"^.*_bs\d+\.o$", path):
                object_files.append(path)
            else:
                continue
            shutil.copy2(os.path.join(aot_dir, path), output_dir)

    n_batch_sizes = len(config["compilation"]["batch_sizes"])
    if len(header_files) != n_batch_sizes:
        raise ValueError(f"expected {n_batch_sizes} header files, got {len(header_files)}")
    if len(object_files) != n_batch_sizes:
        raise ValueError(f"expected {n_batch_sizes} object files, got {len(object_files)}")

    return header_files, object_files


def parse_header(path: str) -> HeaderData:
    # read all non-empty lines
    path = os.path.expandvars(os.path.expanduser(str(path)))
    with open(path, "r") as f:
        lines = [line for line in (line.strip() for line in f.readlines()) if line]

    # prepare HeaderData
    data = HeaderData(*([None] * len(HeaderData._fields)))

    # helper to set data fields
    set_ = lambda key, value: data._replace(**{key: value})

    # extract data
    arg_counts = {}
    res_counts = {}
    while lines:
        line = lines.pop(0)

        # read the namespace
        m = re.match(r"^namespace\s+([^\s]+)\s*\{$", line)
        if m:
            data = set_("namespace", m.group(1))
            continue

        # read the class name and batch size
        m = re.match(rf"^class\s+([^\s]+)_bs(\d+)\s+final\s+\:\s+public\stensorflow\:\:XlaCompiledCpuFunction\s+.*$", line)  # noqa
        if m:
            data = set_("class_name", m.group(1))
            data = set_("batch_size", int(m.group(2)))

        # read argument and result counts
        m = re.match(r"^int\s+(arg|result)(\d+)_count\(\).+$", line)
        if m:
            # get kind and index
            kind = m.group(1)
            index = int(m.group(2))

            # parse the next line
            m = re.match(r"^return\s+(\d+)\s*\;.*$", lines.pop(0))
            if not m:
                raise Exception(f"corrupted header file {path}")
            count = int(m.group(1))

            # store the count
            (arg_counts if kind == "arg" else res_counts)[index] = count
            continue

    # helper to flatten counts to lists
    def flatten(counts: dict[int, int], name: str) -> list[int]:
        if set(counts) != set(range(len(counts))):
            raise ValueError(
                f"non-contiguous indices in {name} counts: {', '.join(map(str, counts))}",
            )
        return [counts[index] for index in sorted(counts)]

    # helper to enforce integer division by batch size
    def no_batch(count: int, index: int, name: str) -> int:
        if count % data.batch_size != 0:
            raise ValueError(
                f"{name} count of {count} at index {index} is not dividable by batch size "
                f"{data.batch_size}",
            )
        return count // data.batch_size

    # store the prefix
    base = os.path.basename(path)
    postfix = f"_bs{data.batch_size}.h"
    if not base.endswith(postfix):
        raise ValueError(f"header '{path}' does not end with expected postfix '{postfix}'")
    data = set_("prefix", base[:-len(postfix)])

    # set counts
    data = set_("n_args", len(arg_counts))
    data = set_("n_res", len(res_counts))
    data = set_("arg_counts", flatten(arg_counts, "argument"))
    data = set_("res_counts", flatten(res_counts, "result"))
    data = set_("arg_counts_no_batch", tuple(
        no_batch(c, i, "argument")
        for i, c in enumerate(data.arg_counts)
    ))
    data = set_("res_counts_no_batch", tuple(
        no_batch(c, i, "result")
        for i, c in enumerate(data.res_counts)
    ))

    return data


def create_wrapper(
    output_file: str,
    header_files: list[str],
    model_dir: str,
    include_guard: str = "tfaot_model",
    template: str = os.path.join(this_dir, "wrapper.h.in"),
) -> None:
    if not header_files:
        raise ValueError("no header files provided")

    # read header data
    header_data = {}
    for path in header_files:
        data = parse_header(path)
        header_data[data.batch_size] = data

    # sorted batch sizes
    batch_sizes = sorted(data.batch_size for data in header_data.values())

    # set common variables
    variables = {
        "model_path": model_dir,
        "batch_sizes": batch_sizes,
        "include_guard": include_guard,
    }
    for key in common_header_data:
        values = set(getattr(d, key) for d in header_data.values())
        if len(values) > 1:
            raise ValueError(f"found more than one possible {key} values: {', '.join(values)}")
        variables[key] = values.pop()

    # helper for variable replacement
    def substituter(variables):
        # insert upper-case variants of strings, csv variants of lists
        variables_ = {}
        for key, value in variables.items():
            key = key.upper()
            variables_[key] = str(value)
            if isinstance(value, str) and not key.endswith("_UC"):
                variables_[f"{key}_UC"] = value.upper()
            elif isinstance(value, (list, tuple)) and not key.endswith("_CSV"):
                variables_[f"{key}_CSV"] = ", ".join(map(str, value))

        def repl(m):
            key = m.group(1)
            if key not in variables_:
                raise KeyError(f"template contains unknown variable {key}")
            return variables_[key]

        return lambda line: re.sub(r"\$\{([A-Z0-9_]+)\}", repl, line)

    # substituter for common variables and per-model variables
    common_sub = substituter(variables)
    model_subs = {
        batch_size: substituter({
            **variables,
            **dict(zip(HeaderData._fields, header_data[batch_size])),
        })
        for batch_size in batch_sizes
    }

    # read template lines
    template = os.path.expandvars(os.path.expanduser(str(template)))
    with open(template, "r") as f:
        input_lines = [line.rstrip() for line in f.readlines()]

    # go through lines and define new ones
    output_lines = []
    while input_lines:
        line = input_lines.pop(0)

        # loop statement?
        m = re.match(r"^\/\/\s+foreach=([^\s]+)\s+lines=(\d+)$", line.strip())
        if m:
            loop = m.group(1)
            n_lines = int(m.group(2))

            if loop == "MODEL":
                # repeat the next n lines for each batch size and replace model variables
                batch_lines, input_lines = input_lines[:n_lines], input_lines[n_lines:]
                for batch_size in batch_sizes:
                    for line in batch_lines:
                        output_lines.append(model_subs[batch_size](line))
            else:
                raise ValueError(f"unknown loop target '{loop}'")

            continue

        # just make common substitutions
        output_lines.append(common_sub(line))

    # prepare the output
    output_file = os.path.expandvars(os.path.expanduser(str(output_file)))
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # write lines
    with open(output_file, "w") as f:
        f.writelines("\n".join(map(str, output_lines)) + "\n")

    return output_file


def create_toolfile(
    output_file: str,
    tool_vars: dict[str, str],
    template: str = os.path.join(this_dir, "toolfile.xml.in"),
) -> str:
    # check mandatory tool variables
    tool_vars = tool_vars or {}
    if not tool_vars.get("tool_name"):
        raise ValueError("missing field 'tool_name' in tool_vars")

    # fill defaults
    default_base_name = tool_vars["tool_name"].upper().replace("-", "_") + "_BASE"
    tool_vars.setdefault("tool_base_name", default_base_name)
    tool_vars.setdefault("tool_version", "1.0.0")
    tool_vars.setdefault("tool_base", "@TOOL_BASE@")
    tool_vars.setdefault("lib_dir", "lib")
    tool_vars.setdefault("inc_dir", "include")
    tool_vars.setdefault("ld_flags", [])

    # formatting
    if isinstance(tool_vars["ld_flags"], list):
        ld_flags = []
        for flag in tool_vars["ld_flags"]:
            # when only a basename was given, prepend the tool base
            if "/" not in flag and "<" not in flag:
                flag = f"${tool_vars['tool_base_name']}/{tool_vars['lib_dir']}/{flag}"
            # insert into flags tag
            if not flag.startswith("<flags "):
                flag = f"<flags LDFLAGS=\"{flag}\"/>"
            ld_flags.append(flag)
        tool_vars["ld_flags"] = "\n  ".join(ld_flags)

    # read template lines
    template = os.path.expandvars(os.path.expanduser(str(template)))
    with open(template, "r") as f:
        input_lines = [line.rstrip() for line in f.readlines()]

    # prepare the output
    output_file = os.path.expandvars(os.path.expanduser(str(output_file)))
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # write lines
    with open(output_file, "w") as f:
        for line in input_lines:
            f.write(f"{line.format(**tool_vars)}\n")

    return output_file

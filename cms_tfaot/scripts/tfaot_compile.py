#!/usr/bin/env python3
# coding: utf-8

"""
Script that takes an AOT configuration file, compiles the referenced model and provides all files
necessary for the production deployment.
"""

from __future__ import annotations

import os
import re
import shutil
import platform
import collections
from typing import Any


CompilationResult = collections.namedtuple(
    "CompilationResult",
    [
        "output_dir",
        "header_dir",
        "header_files",
        "wrapper_file",
        "object_dir",
        "object_files",
        "tool_file",
        "tool_name",
    ],
)


def tfaot_compile(
    config_file: str,
    output_dir: str,
    tool_name: str | None = None,
    tool_base: str | None = None,
    dev: bool = False,
    additional_flags: list[str] | str | None = None,
) -> CompilationResult:
    # deferred imports
    from cms_tfaot import load_and_normalize_config, compile_model, create_wrapper, create_toolfile

    # prepare output directory
    output_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(output_dir)))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load and normalize the config
    config = load_and_normalize_config(config_file)

    # default tool name
    if not tool_name:
        tool_name = f"tfaot-model-{config['model']['name'].replace('_', '-')}"
    if not tool_base:
        tool_base = cmssw_rel_path(output_dir) if dev else "@TOOL_BASE@"

    # compile
    header_files, object_files = compile_model(
        config,
        output_dir,
        additional_flags=additional_flags,
    )

    # create subdirectories and move files in dev mode
    header_dir = output_dir
    object_dir = output_dir
    if dev:
        # header files
        header_dir = os.path.join(output_dir, "include", tool_name)
        if os.path.exists(header_dir):
            shutil.rmtree(header_dir)
        os.makedirs(header_dir)
        for h in header_files:
            shutil.move(os.path.join(output_dir, h), header_dir)
        header_files = [os.path.join("include", tool_name, h) for h in header_files]
        # object files
        object_dir = os.path.join(output_dir, "lib")
        if os.path.exists(object_dir):
            shutil.rmtree(object_dir)
        os.makedirs(object_dir)
        for o in object_files:
            shutil.move(os.path.join(output_dir, o), object_dir)
        object_files = [os.path.join("lib", o) for o in object_files]

    # create the header wrapper
    wrapper_file = create_wrapper(
        output_file=os.path.join(header_dir, f"{config['model']['name']}.h"),
        header_files=[os.path.join(output_dir, h) for h in header_files],
        model_dir=config["model"]["saved_model"],
    )

    # create a symlink to the model header
    os.symlink(os.path.basename(wrapper_file), os.path.join(header_dir, "model.h"))

    # create the toolfile
    tool_file = create_toolfile(
        output_file=os.path.join(output_dir, f"{tool_name}.xml"),
        tool_vars={
            "tool_name": tool_name,
            "tool_version": config["model"]["version"],
            "tool_base": tool_base,
            "lib_dir": "lib",
            "inc_dir": "include",
            "ld_flags": list(map(os.path.basename, object_files)),
        },
    )

    result = CompilationResult(
        output_dir=output_dir,
        header_dir=header_dir,
        header_files=header_files,
        wrapper_file=wrapper_file,
        object_dir=object_dir,
        object_files=object_files,
        tool_file=tool_file,
        tool_name=tool_name,
    )

    # print some information in dev mode
    if dev:
        print_compilation_info(config, result)


def cmssw_rel_path(path: str) -> str:
    cmssw_base = os.getenv("CMSSW_BASE")
    if not cmssw_base:
        return path

    rel = os.path.relpath(path, cmssw_base)
    if rel.startswith(".."):
        return path

    return f"$CMSSW_BASE/{rel}"


def print_compilation_info(config: dict[str, Any], result: CompilationResult) -> None:
    model_name = config["model"]["name"]
    class_name = f"{config['compilation']['namespace']}::{model_name}"
    batch_sizes_str = ",".join(map(str, config["compilation"]["batch_sizes"]))

    print(f"\n{80 * '-'}")
    print(f"\nsuccessfully AOT compiled model '{model_name}' for batch sizes: {batch_sizes_str}")
    print("\n  1. register it to scram:")
    print(f"     > scram setup {cmssw_rel_path(result.tool_file)}")
    print("\n  2. 'use' the tool in your BuildFile.xml:")
    print(f"     <use name=\"{result.tool_name}.xml\"/>")
    print("\n  3. include the following header in your code:")
    print(f"     #include \"{result.tool_name}/model.h\"")
    print("\n  4. create an AOT model instance via:")
    print(f"     auto model = tfaot::Model<{class_name}>();\n")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aot-config",
        "-c",
        help="aot configuration file (yaml or json)",
        required=True,
    )
    parser.add_argument(
        "--output-directory",
        "-o",
        help="output directory for compile targets",
        required=True,
    )
    parser.add_argument(
        "--tool-name",
        help="name of the tool; defaults to 'tfaot-model-<model-name>'",
        default=None,
    )
    parser.add_argument(
        "--tool-base",
        help="base directory of the tool; no default",
        default=None,
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="activates the development workflow, setting some variables to sensible defaults",
    )
    parser.add_argument(
        "--additional-flags",
        help="additional flags to be passed to the underlying aot compiler invocation",
    )
    args = parser.parse_args()

    # check if the platform arch matches the "--target_triple" value in additional_flags
    if args.dev:
        m = re.match(r"^.*--target_triple(\s+|=)([^-]+)-.+$", args.additional_flags or "")
        triple_arch = m.group(2) if m else "x86_64"
        arch = platform.processor()
        if triple_arch != arch:
            msg = f"\nWARNING: your platform architecture is '{arch}'"
            if m:
                msg += f" which does not match the configured '--target_triple' of '{triple_arch}'"
            else:
                msg += " but the default compilation target is 'x86_64'"
            msg += (
                ", which can lead to unexpected behavior in the compiled model, so please set the"
                " correct target via --additional-flags=\"--target_triple=<arch>-unknown-linux\"\n"
            )
            print(msg)

    tfaot_compile(
        config_file=args.aot_config,
        output_dir=args.output_directory,
        tool_name=args.tool_name,
        tool_base=args.tool_base,
        dev=args.dev,
        additional_flags=args.additional_flags,
    )

    return 0


if __name__ == "__main__":
    main()

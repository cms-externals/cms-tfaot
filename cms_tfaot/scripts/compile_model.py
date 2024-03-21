#!/usr/bin/env python3
# coding: utf-8

"""
Script that takes an AOT configuration file, compiles the referenced model and provides all files
necessary for the production deployment.
"""

from __future__ import annotations

import os
import collections


CompilationResult = collections.namedtuple(
    "CompilationResult",
    ["output_dir", "header_files", "object_files", "wrapper_file", "tool_file"],
)


def compile_model(
    config_file: str,
    output_dir: str,
    tool_name: str | None = None,
    tool_base: str | None = None,
) -> CompilationResult:
    # deferred imports
    from cms_tfaot import load_and_normalize_config, compile_model, create_wrapper, create_toolfile

    # prepare output directory
    output_dir = os.path.abspath(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load and normalize the config
    config = load_and_normalize_config(config_file)

    # compile
    header_files, object_files = compile_model(config, output_dir)

    # create the header wrapper
    wrapper_file = create_wrapper(
        output_file=os.path.join(output_dir, f"{config['model']['name']}.h"),
        header_files=[os.path.join(output_dir, h) for h in header_files],
        model_dir=config["model"]["saved_model"],
    )

    # create the toolfile
    if not tool_name:
        tool_name = f"tfaot-model-{config['model']['name'].replace('_', '-')}"
    tool_file = create_toolfile(
        output_file=os.path.join(output_dir, f"{tool_name}.xml"),
        tool_vars={
            "tool_name": tool_name,
            "tool_version": config["model"]["version"],
            "tool_base": tool_base or "@TOOL_BASE@",
            "lib_dir": "lib",
            "inc_dir": "include",
            "ld_flags": list(map(os.path.basename, object_files)),
        },
    )

    return CompilationResult(
        output_dir=output_dir,
        header_files=header_files,
        object_files=object_files,
        wrapper_file=wrapper_file,
        tool_file=tool_file,
    )


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
    args = parser.parse_args()

    compile_model(
        config_file=args.aot_config,
        output_dir=args.output_directory,
        tool_name=args.tool_name,
        tool_base=args.tool_base,
    )

    return 0


if __name__ == "__main__":
    main()

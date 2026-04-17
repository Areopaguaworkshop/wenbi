#!/usr/bin/env python3
"""Main CLI entry point for cli-anything-list.

Discovers and lists all CLI-Anything tools (installed and generated).
"""
import os
import sys
import click

from cli_anything.list.core.scanner import scan_installed, scan_generated, merge_tools
from cli_anything.list.core.export import format_json, format_table


@click.command()
@click.option("--path", default=".", type=click.Path(exists=True),
              help="Directory to search for generated CLIs (default: current directory).")
@click.option("--depth", type=int, default=None,
              help="Maximum recursion depth for scanning (default: unlimited). "
                   "Use 0 for current directory only, 1 for one level deep, etc.")
@click.option("--json", "json_mode", is_flag=True, default=False,
              help="Output in JSON format for machine parsing.")
def cli(path, depth, json_mode):
    """List all available CLI-Anything tools (installed and generated).

    Scans for installed pip packages named 'cli-anything-*' and local
    generated directories matching the agent-harness/cli_anything/* pattern.

    Use --json for machine-readable output.
    Use --path to specify a search directory.
    Use --depth to limit recursion depth.
    """
    if not os.path.exists(path):
        click.echo(f"Error: Path does not exist: {path}", err=True)
        sys.exit(1)

    # Scan installed packages
    installed = scan_installed()

    # Scan generated directories
    generated = scan_generated(search_path=path, depth=depth)

    # Merge results
    tools = merge_tools(installed, generated)

    # Format and output
    if json_mode:
        click.echo(format_json(tools))
    else:
        click.echo(format_table(tools))


def main():
    cli()


if __name__ == "__main__":
    main()
"""Output formatting for cli-anything-list."""
import json


def format_json(tools):
    """Format tools dict as JSON string.

    Args:
        tools: Dict of tool info from merge_tools().

    Returns:
        str: Pretty-printed JSON.
    """
    installed_count = sum(1 for t in tools.values() if t["status"] == "installed")
    generated_only = sum(1 for t in tools.values() if t["status"] == "generated")

    output = {
        "tools": [
            {"name": name, **info}
            for name, info in sorted(tools.items())
        ],
        "total": len(tools),
        "installed": installed_count,
        "generated_only": generated_only,
    }
    return json.dumps(output, indent=2)


def format_table(tools):
    """Format tools dict as an aligned text table.

    Args:
        tools: Dict of tool info from merge_tools().

    Returns:
        str: Formatted table string.
    """
    if not tools:
        return "CLI-Anything Tools (found 0)\n\nNo tools found."

    lines = []
    lines.append(f"CLI-Anything Tools (found {len(tools)})")
    lines.append("")

    # Calculate column widths
    name_w = max(len(n) for n in tools) + 2
    status_w = 12
    version_w = 10
    source_w = 40

    # Min widths
    name_w = max(name_w, len("Name") + 2)
    source_w = max(source_w, len("Source") + 2)

    # Header
    header = f"{'Name':<{name_w}} {'Status':<{status_w}} {'Version':<{version_w}} Source"
    lines.append(header)
    lines.append("─" * len(header))

    # Rows
    for name in sorted(tools):
        info = tools[name]
        version = info.get("version") or "-"
        source = info.get("source") or "-"
        status = info["status"]
        lines.append(f"{name:<{name_w}} {status:<{status_w}} {version:<{version_w}} {source}")

    return "\n".join(lines)
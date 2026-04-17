# cli-anything-list

Discover and list all CLI-Anything tools available on your system.

## Installation

```bash
cd agent-harness
pip install -e .
```

This installs both `cli-anything-wenbi` and `cli-anything-list`.

## Usage

### List all tools (default)

```bash
cli-anything-list
```

Output:
```
CLI-Anything Tools (found 2)

Name    Status       Version    Source
──────────────────────────────────────
list    generated    0.1.0      agent-harness/cli_anything
wenbi   installed    0.1.0      agent-harness/cli_anything
```

### JSON output (for agent consumption)

```bash
cli-anything-list --json
```

```json
{
  "tools": [
    {"name": "wenbi", "status": "installed", "version": "0.1.0", "executable": "/usr/local/bin/cli-anything-wenbi", "source": "..."}
  ],
  "total": 1,
  "installed": 1,
  "generated_only": 0
}
```

### Search a specific directory

```bash
cli-anything-list --path /path/to/project
```

### Limit recursion depth

```bash
cli-anything-list --depth 0    # Current directory only
cli-anything-list --depth 1    # One level deep
cli-anything-list              # Unlimited (default)
```

## How It Works

1. **Installed CLIs**: Scans `importlib.metadata` for packages named `cli-anything-*`
2. **Generated CLIs**: Globs for `agent-harness/cli_anything/*/__init__.py` patterns
3. **Merges results**: Deduplicates by software name; installed status takes priority

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--path` | `.` | Directory to search for generated CLIs |
| `--depth` | unlimited | Maximum recursion depth for scanning |
| `--json` | false | Output in JSON format |
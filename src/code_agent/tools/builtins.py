"""Built-in tools for file operations and code search.

These are the core tools that most coding agents need. They mirror
what Claude Code, OpenCode, and other agents provide.

Tools:
- read_file: Read file contents
- write_file: Write/create files
- glob_files: Find files by pattern
- grep_search: Search file contents
- list_directory: List directory contents
"""

import fnmatch
import os
import re
from pathlib import Path

from code_agent.tools.registry import tool


@tool(
    name="read_file",
    description="Read the contents of a file. Returns the file content as a string.",
    parameters={
        "path": {"type": "string", "description": "Path to the file to read"},
    },
)
def read_file(path: str) -> str:
    """Read contents of a file."""
    try:
        return Path(path).read_text()
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


@tool(
    name="write_file",
    description="Write content to a file. Creates the file if it doesn't exist, overwrites if it does.",
    parameters={
        "path": {"type": "string", "description": "Path to the file to write"},
        "content": {"type": "string", "description": "Content to write to the file"},
    },
)
def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Successfully wrote {len(content)} characters to {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool(
    name="glob_files",
    description="Find files matching a glob pattern (e.g., '**/*.py' for all Python files).",
    parameters={
        "pattern": {"type": "string", "description": "Glob pattern to match files"},
        "directory": {"type": "string", "description": "Directory to search in (default: current)"},
    },
)
def glob_files(pattern: str, directory: str = ".") -> str:
    """Find files matching a glob pattern."""
    try:
        base_path = Path(directory)
        matches = list(base_path.glob(pattern))
        if not matches:
            return f"No files found matching pattern: {pattern}"
        return "\n".join(str(m) for m in sorted(matches)[:100])  # Limit to 100 results
    except Exception as e:
        return f"Error searching files: {e}"


@tool(
    name="grep_search",
    description="Search for a pattern in files. Returns matching lines with file paths and line numbers.",
    parameters={
        "pattern": {"type": "string", "description": "Regex pattern to search for"},
        "path": {"type": "string", "description": "File or directory to search in"},
        "file_pattern": {"type": "string", "description": "Glob pattern to filter files (e.g., '*.py')"},
    },
)
def grep_search(pattern: str, path: str = ".", file_pattern: str = "*") -> str:
    """Search for a pattern in files."""
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Invalid regex pattern: {e}"

    results = []
    search_path = Path(path)

    try:
        if search_path.is_file():
            files = [search_path]
        else:
            files = [f for f in search_path.rglob("*") if f.is_file() and fnmatch.fnmatch(f.name, file_pattern)]

        for file_path in files[:50]:  # Limit files searched
            try:
                content = file_path.read_text(errors="ignore")
                for line_num, line in enumerate(content.splitlines(), 1):
                    if regex.search(line):
                        results.append(f"{file_path}:{line_num}: {line.strip()[:100]}")
                        if len(results) >= 50:  # Limit results
                            results.append("... (truncated)")
                            return "\n".join(results)
            except (PermissionError, IsADirectoryError):
                continue

        if not results:
            return f"No matches found for pattern: {pattern}"
        return "\n".join(results)

    except Exception as e:
        return f"Error searching: {e}"


@tool(
    name="list_directory",
    description="List contents of a directory with file types and sizes.",
    parameters={
        "path": {"type": "string", "description": "Directory path to list"},
    },
)
def list_directory(path: str = ".") -> str:
    """List directory contents."""
    try:
        dir_path = Path(path)
        if not dir_path.is_dir():
            return f"Error: Not a directory: {path}"

        entries = []
        for entry in sorted(dir_path.iterdir()):
            if entry.is_dir():
                entries.append(f"[DIR]  {entry.name}/")
            else:
                size = entry.stat().st_size
                size_str = _format_size(size)
                entries.append(f"[FILE] {entry.name} ({size_str})")

        if not entries:
            return f"Directory is empty: {path}"
        return "\n".join(entries[:100])  # Limit to 100 entries

    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error listing directory: {e}"


def _format_size(size: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


# Convenience function to get all built-in tools
def get_builtin_tools() -> list:
    """Return list of all built-in tool functions."""
    return [read_file, write_file, glob_files, grep_search, list_directory]

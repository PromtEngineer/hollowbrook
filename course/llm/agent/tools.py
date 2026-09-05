"""Tools: what an agent is allowed to *do* (Chapters 24 and 26).

A **tool** is a plain Python function plus a description the model can read. The
description is written as a JSON Schema, the same format the Anthropic and OpenAI
APIs and MCP all use, so the model learns one convention.

Three rules make tools agent-friendly:

1. **Errors are observations.** ``ToolRegistry.call`` never raises. A wrong argument
   or a missing file comes back as *text* and the model gets to read it and adapt,
   exactly like a person reading an error message.
2. **Tools are sandboxed.** Every built-in file tool refuses paths that escape the
   working directory. The model may be confused or adversarial; the harness must not be.
3. **Read-only is a first-class flag.** Permission policies (harness.py) can say
   "allow anything that only reads" — so each tool declares what it touches.
"""
from __future__ import annotations

import ast
import json
import operator
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Tool:
    """One callable the model may invoke.

    ``parameters`` is a JSON Schema object: ``{"type": "object", "properties": {...},
    "required": [...]}``. ``fn`` takes the arguments as keywords and returns a string
    (the model only ever sees text).
    """
    name: str
    description: str
    parameters: dict
    fn: Callable[..., str]
    read_only: bool = True

    def schema(self) -> dict:
        """The wire format shared by the Anthropic API and MCP ("input_schema")."""
        return {"name": self.name, "description": self.description, "input_schema": self.parameters}


class ToolRegistry:
    """A name -> Tool map with schema export and *safe* calling."""

    def __init__(self, tools: Optional[list[Tool]] = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def schemas(self) -> list[dict]:
        """What the model sees: [{name, description, input_schema}, ...]."""
        return [t.schema() for t in self._tools.values()]

    def subset(self, names: list[str]) -> "ToolRegistry":
        """A new registry with only ``names`` (used for sub-agents and planning)."""
        return ToolRegistry([self._tools[n] for n in names if n in self._tools])

    def validate(self, tool: Tool, args: dict) -> Optional[str]:
        """Return an error string if ``args`` violate the schema, else None.

        Only the parts of JSON Schema a beginner needs: required keys, unknown keys,
        and the basic scalar types. Real harnesses use a full validator.
        """
        if not isinstance(args, dict):
            return f"arguments must be a JSON object, got {type(args).__name__}"
        props = tool.parameters.get("properties", {})
        for key in tool.parameters.get("required", []):
            if key not in args:
                return f"missing required argument '{key}' (expected: {sorted(props)})"
        for key, value in args.items():
            if key not in props:
                return f"unknown argument '{key}' (expected: {sorted(props)})"
            want = props[key].get("type")
            if want and not _json_type_ok(value, want):
                return f"argument '{key}' should be of type {want}, got {type(value).__name__}"
        return None

    def call(self, name: str, args: dict) -> str:
        """Run a tool and return its text. Never raises: errors become observations."""
        tool = self.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'. Available tools: {self.names()}"
        problem = self.validate(tool, args)
        if problem:
            return f"Error calling {name}: {problem}"
        try:
            out = tool.fn(**args)
            return out if isinstance(out, str) else json.dumps(out)
        except Exception as e:  # noqa: BLE001 — the whole point is to catch everything
            return f"Error calling {name}: {type(e).__name__}: {e}"


_JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
               "object": dict, "array": list}


def _json_type_ok(value: Any, want: str) -> bool:
    py = _JSON_TYPES.get(want)
    if py is None:
        return True
    if want in ("integer", "number") and isinstance(value, bool):
        return False  # bool is an int in Python but not in JSON
    return isinstance(value, py)


# ------------------------------------------------------------- built-in tools
def _params(props: dict[str, tuple[str, str]], required: list[str]) -> dict:
    """Tiny helper: {"path": ("string", "File to read")} -> JSON Schema."""
    return {
        "type": "object",
        "properties": {k: {"type": t, "description": d} for k, (t, d) in props.items()},
        "required": required,
    }


class _Sandbox:
    """Resolves model-supplied paths inside ``workdir`` and refuses escapes."""

    def __init__(self, workdir: str) -> None:
        self.root = os.path.realpath(workdir)
        os.makedirs(self.root, exist_ok=True)

    def resolve(self, path: str) -> str:
        full = os.path.realpath(os.path.join(self.root, path))
        if full != self.root and not full.startswith(self.root + os.sep):
            raise PermissionError(f"path '{path}' escapes the sandbox {self.root}")
        return full


# Only these AST nodes may appear in a calculator expression. Anything else
# (names, calls, attributes — the way ``__import__('os')`` sneaks in) is rejected.
_SAFE_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
             ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
             ast.UAdd: operator.pos, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv}


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic (+ - * / ** % //, parentheses) without ``eval``'s dangers."""
    def walk(node: ast.AST):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("exponent too large")  # 9**9**9 would hang the process
            return _SAFE_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](walk(node.operand))
        raise ValueError(f"unsupported syntax: {type(node).__name__}")
    tree = ast.parse(expression.strip(), mode="eval")
    return walk(tree)


def make_builtin_tools(workdir: str) -> ToolRegistry:
    """The small tool set a coding agent needs, sandboxed to ``workdir``."""
    box = _Sandbox(workdir)
    reg = ToolRegistry()

    def read_file(path: str) -> str:
        with open(box.resolve(path), encoding="utf-8", errors="replace") as f:
            return f.read()

    def write_file(path: str, content: str) -> str:
        full = box.resolve(path)
        os.makedirs(os.path.dirname(full) or box.root, exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} chars to {path}"

    def list_dir(path: str = ".") -> str:
        full = box.resolve(path)
        entries = sorted(os.listdir(full))
        return "\n".join(e + ("/" if os.path.isdir(os.path.join(full, e)) else "") for e in entries) or "(empty)"

    def search(pattern: str, path: str = ".") -> str:
        rx = re.compile(pattern)
        hits: list[str] = []
        for dirpath, dirnames, filenames in os.walk(box.resolve(path)):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__pycache__"]
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                try:
                    with open(fp, encoding="utf-8") as f:
                        for i, line in enumerate(f, 1):
                            if rx.search(line):
                                hits.append(f"{os.path.relpath(fp, box.root)}:{i}: {line.rstrip()}")
                except (UnicodeDecodeError, OSError):
                    continue
                if len(hits) >= 100:
                    return "\n".join(hits) + "\n... (truncated at 100 hits)"
        return "\n".join(hits) or "(no matches)"

    def calculator(expression: str) -> str:
        return str(safe_eval(expression))

    def run_python(code: str) -> str:
        p = subprocess.run([sys.executable, "-c", code], cwd=box.root, capture_output=True,
                           text=True, timeout=10)
        out = p.stdout + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
        return (out.strip() or "(no output)") + f"\n[exit code {p.returncode}]"

    def run_tests(path: str = "tests") -> str:
        try:
            p = subprocess.run([sys.executable, "-m", "pytest", "-q", path], cwd=box.root,
                               capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            return "FAIL: tests timed out after 120 s"
        lines = (p.stdout + p.stderr).strip().splitlines()
        status = "PASS" if p.returncode == 0 else "FAIL"
        return f"{status} (exit code {p.returncode})\n" + "\n".join(lines[-30:])

    reg.register(Tool("read_file", "Read a UTF-8 text file inside the workdir.",
                      _params({"path": ("string", "Path relative to the workdir")}, ["path"]), read_file))
    reg.register(Tool("write_file", "Create or overwrite a text file inside the workdir.",
                      _params({"path": ("string", "Path relative to the workdir"),
                               "content": ("string", "Full new file content")}, ["path", "content"]),
                      write_file, read_only=False))
    reg.register(Tool("list_dir", "List the entries of a directory (directories end with '/').",
                      _params({"path": ("string", "Directory, default '.'")}, []), list_dir))
    reg.register(Tool("search", "Regex search over all text files under a directory (like grep -rn).",
                      _params({"pattern": ("string", "Python regular expression"),
                               "path": ("string", "Directory to search, default '.'")}, ["pattern"]), search))
    reg.register(Tool("calculator", "Evaluate an arithmetic expression (+ - * / ** and parentheses).",
                      _params({"expression": ("string", "e.g. '(2 + 3) * 4'")}, ["expression"]), calculator))
    reg.register(Tool("run_python", "Run a Python snippet in the workdir (10 s timeout) and return its output.",
                      _params({"code": ("string", "Python source code")}, ["code"]), run_python, read_only=False))
    reg.register(Tool("run_tests", "Run `python -m pytest -q` in the workdir; returns PASS/FAIL and the last lines.",
                      _params({"path": ("string", "Test file or directory, default 'tests'")}, []), run_tests))
    return reg

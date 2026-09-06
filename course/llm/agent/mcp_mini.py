"""A minimal Model Context Protocol (MCP) server and client (Chapter 26).

MCP is the standard that lets *any* agent use *any* tool server. Under the hood it
is **JSON-RPC 2.0** — request ``{"jsonrpc": "2.0", "id": 1, "method": ..., "params": ...}``,
response ``{"jsonrpc": "2.0", "id": 1, "result": ...}`` — carried over a transport.
We implement the simplest transport, **stdio**: the client starts the server as a
subprocess and they exchange one JSON object per line.

What this file implements (enough to plug real MCP tools into our Agent):

    initialize                 handshake: protocol version, capabilities, server name
    notifications/initialized  the client's "I'm ready" (a notification: no id, no reply)
    tools/list                 tools with name / description / inputSchema
    tools/call                 run one tool -> {content: [{type: "text", text}], isError}
    ping                       liveness check

What the full 2025-11-25 spec adds and we leave out on purpose:

* **resources** (read-only data the server exposes by URI, like files or DB rows),
* **prompts** (reusable prompt templates the server offers),
* **sampling** (the *server* asks the *client's* model to generate text),
* **elicitation** (the server asks the human a question mid-tool-call),
* the **Streamable HTTP** transport (one HTTP endpoint, optional SSE streaming),
  pagination cursors, progress notifications, logging, and OAuth authorisation.

The shapes of the messages are the real ones, so a client written here can talk to
a real MCP server's tools, and a real client can list and call ours.

Run the built-in tools as a server:

    python3 -m llm.agent.mcp_mini --serve --workdir /path/to/sandbox
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Iterable, Any, Optional, TextIO

from .tools import Tool, ToolRegistry

PROTOCOL_VERSION = "2025-11-25"


# ---------------------------------------------------------------- JSON-RPC
def _response(id_: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _error(id_: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


class MCPServer:
    """Expose a ``ToolRegistry`` over MCP. One JSON-RPC message per line on stdio."""

    def __init__(self, registry: ToolRegistry, name: str = "llm-agent-mini", version: str = "0.1") -> None:
        self.registry = registry
        self.name, self.version = name, version

    def handle(self, msg: dict) -> Optional[dict]:
        """Dispatch one request. Returns the response, or None for notifications."""
        method, params, id_ = msg.get("method"), msg.get("params") or {}, msg.get("id")
        if method == "initialize":
            return _response(id_, {"protocolVersion": PROTOCOL_VERSION,
                                   "capabilities": {"tools": {}},
                                   "serverInfo": {"name": self.name, "version": self.version}})
        if method == "notifications/initialized":
            return None                                          # notifications get no reply
        if method == "ping":
            return _response(id_, {})
        if method == "tools/list":
            tools = [{"name": t["name"], "description": t["description"], "inputSchema": t["input_schema"],
                      # readOnlyHint is the MCP annotation a client may use for its permission gate
                      "annotations": {"readOnlyHint": self.registry.get(t["name"]).read_only}}
                     for t in self.registry.schemas()]
            return _response(id_, {"tools": tools})
        if method == "tools/call":
            name, args = params.get("name", ""), params.get("arguments") or {}
            if name not in self.registry:
                return _error(id_, -32602, f"Unknown tool: {name}")   # protocol-level error
            text = self.registry.call(name, args)
            # tool-level failures are *results* with isError=True, so the model sees them
            return _response(id_, {"content": [{"type": "text", "text": text}],
                                   "isError": text.startswith("Error")})
        if id_ is None:
            return None                                          # unknown notification: ignore
        return _error(id_, -32601, f"Method not found: {method}")

    def serve(self, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
        """Read requests line by line until stdin closes."""
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                out: Optional[dict] = _error(None, -32700, "Parse error")
            else:
                out = self.handle(msg)
            if out is not None:
                stdout.write(json.dumps(out) + "\n")
                stdout.flush()


class MCPClient:
    """Start an MCP server as a subprocess and talk JSON-RPC to it over its pipes."""

    def __init__(self, cmd: list[str], cwd: Optional[str] = None, env: Optional[dict] = None) -> None:
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, cwd=cwd, env=env, bufsize=1)
        self._next_id = 0
        self.server_info: dict = {}

    def _send(self, method: str, params: Optional[dict] = None, notify: bool = False) -> Any:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if not notify:
            self._next_id += 1
            msg["id"] = self._next_id
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        if notify:
            return None
        line = self.proc.stdout.readline()
        if not line:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"MCP server closed the connection. stderr:\n{err}")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(f"MCP error {reply['error'].get('code')}: {reply['error'].get('message')}")
        return reply["result"]

    def initialize(self) -> dict:
        """The handshake every MCP session starts with."""
        self.server_info = self._send("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "llm-agent-client", "version": "0.1"},
        })
        self._send("notifications/initialized", notify=True)
        return self.server_info

    def ping(self) -> bool:
        return self._send("ping") == {}

    def list_tools(self) -> list[dict]:
        return self._send("tools/list")["tools"]

    def call_tool(self, name: str, arguments: Optional[dict] = None) -> str:
        """Returns the concatenated text content (our tools only emit text)."""
        result = self._send("tools/call", {"name": name, "arguments": arguments or {}})
        return "\n".join(b.get("text", "") for b in result.get("content", []) if b.get("type") == "text")

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()

    def __enter__(self) -> "MCPClient":
        self.initialize()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def mcp_tools_to_registry(client: MCPClient, read_only: bool = False,
                          read_only_tools: Optional[Iterable[str]] = None) -> ToolRegistry:
    """Wrap a server's tools so ``Agent`` can use them like local ones.

    A tool is marked read-only if the server says so (``annotations.readOnlyHint``,
    part of the MCP spec), or if its name is in ``read_only_tools``, or if
    ``read_only=True`` for everything. The default is the conservative choice for a
    permission gate: nothing is trusted as read-only unless declared.
    """
    reg = ToolRegistry()
    safe = set(read_only_tools or ())
    for t in client.list_tools():
        def fn(_name=t["name"], **kwargs) -> str:      # bind the name per tool
            return client.call_tool(_name, kwargs)
        hint = bool((t.get("annotations") or {}).get("readOnlyHint", False))
        reg.register(Tool(t["name"], t.get("description", ""), t.get("inputSchema", {"type": "object"}),
                          fn, read_only=read_only or hint or t["name"] in safe))
    return reg


def main(argv: Optional[list[str]] = None) -> None:
    import argparse
    from .tools import make_builtin_tools
    ap = argparse.ArgumentParser(description="Serve the built-in tools over MCP (stdio).")
    ap.add_argument("--serve", action="store_true", help="run the server loop on stdin/stdout")
    ap.add_argument("--workdir", default=os.getcwd(), help="sandbox directory for the file tools")
    args = ap.parse_args(argv)
    if not args.serve:
        ap.print_help()
        return
    # stdout is the protocol channel: never print() anything else to it.
    MCPServer(make_builtin_tools(args.workdir)).serve(sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()

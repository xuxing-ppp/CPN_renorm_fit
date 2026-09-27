from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import fingerprint
from .storage import atomic_copy, atomic_json


STATE_SCHEMA = 1
STAGES = ("pilot", "two_plaq", "observable", "one_plaq", "topology")
ACTION_RANK = {"reuse": 0, "reprocess": 1, "resimulate": 2}


def effective_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the persisted configuration, excluding transient CLI metadata."""
    return {key: value for key, value in cfg.items() if not key.startswith("_")}


def config_fingerprint(cfg: dict[str, Any]) -> str:
    return fingerprint({"schema": STATE_SCHEMA, "config": effective_config(cfg)})


def workspace_dir(cfg: dict[str, Any], output_root: str | Path) -> Path:
    return Path(output_root).resolve() / Path(cfg["_config_path"]).stem


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    out: dict[str, Any] = {}
    for key in sorted(value):
        path = f"{prefix}.{key}" if prefix else str(key)
        out.update(_flatten(value[key], path))
    return out


def _changes(old: dict[str, Any], new: dict[str, Any]) -> list[dict[str, Any]]:
    before, after = _flatten(old), _flatten(new)
    missing = "<missing>"
    return [
        {"path": path, "old": before.get(path, missing),
         "new": after.get(path, missing)}
        for path in sorted(set(before) | set(after))
        if before.get(path, missing) != after.get(path, missing)
    ]


def _raise_action(actions: dict[str, str], stages: tuple[str, ...], action: str) -> None:
    for stage in stages:
        if ACTION_RANK[action] > ACTION_RANK[actions[stage]]:
            actions[stage] = action


def affected_stages(changes: list[dict[str, Any]]) -> dict[str, str]:
    """Conservatively classify work invalidated by effective-config changes."""
    actions = {stage: "reuse" for stage in STAGES}
    for change in changes:
        path = change["path"]
        if path.startswith(("model.", "renormalization.", "runtime.", "hmc.")):
            _raise_action(actions, STAGES, "resimulate")
        elif path.startswith(("pilot.", "chains.pilot", "geometry.")):
            _raise_action(actions, STAGES, "resimulate")
        elif path == "steps.two_plaq.p0":
            _raise_action(actions, ("two_plaq", "observable", "topology"),
                          "reprocess")
        elif path.startswith(("steps.two_plaq.", "chains.two_plaq")):
            _raise_action(actions, ("two_plaq", "observable", "topology"),
                          "resimulate")
        elif path.startswith("scan.beta1."):
            _raise_action(actions, ("observable", "topology"), "reprocess")
        elif path.startswith(("steps.observable.", "chains.observable")):
            _raise_action(actions, ("observable", "topology"), "resimulate")
        elif path == "steps.one_plaq.p0":
            _raise_action(actions, ("one_plaq", "topology"), "reprocess")
        elif path.startswith(("steps.one_plaq.", "chains.one_plaq")):
            _raise_action(actions, ("one_plaq", "topology"), "resimulate")
        elif path.startswith("scan.alpha."):
            _raise_action(actions, ("topology",), "reprocess")
        elif path.startswith(("steps.topo.", "chains.topo")):
            _raise_action(actions, ("topology",), "resimulate")
        elif path.startswith("diagnostics."):
            _raise_action(actions, ("two_plaq", "observable", "one_plaq", "topology"),
                          "reprocess")
        else:
            _raise_action(actions, STAGES, "resimulate")
    return actions


@dataclass(frozen=True)
class WorkspaceChange:
    run_dir: Path
    source: str
    config: dict[str, Any]
    config_fingerprint: str
    changes: list[dict[str, Any]]
    source_changed: bool
    previous_source: str | None
    actions: dict[str, str]

    @property
    def needs_confirmation(self) -> bool:
        return bool(self.changes or self.source_changed)


def inspect_workspace(cfg: dict[str, Any], output_root: str | Path, *,
                      baseline: WorkspaceChange | None = None) -> WorkspaceChange:
    run_dir = workspace_dir(cfg, output_root)
    state_path = run_dir / "run_state.json"
    previous: dict[str, Any] = {}
    if baseline is not None:
        previous = {"config": baseline.config, "source": baseline.source}
    elif state_path.exists():
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"cannot read workspace state {state_path}") from exc
        if previous.get("schema") != STATE_SCHEMA:
            raise RuntimeError(
                f"unsupported workspace state schema in {state_path}")
    current = effective_config(cfg)
    source = str(Path(cfg["_config_path"]).resolve())
    old_config = previous.get("config")
    changes = _changes(old_config, current) if isinstance(old_config, dict) else []
    old_source = previous.get("source")
    return WorkspaceChange(
        run_dir=run_dir, source=source, config=current,
        config_fingerprint=config_fingerprint(cfg),
        changes=changes,
        source_changed=bool(old_source and old_source != source),
        previous_source=str(old_source) if old_source else None,
        actions=affected_stages(changes),
    )


def format_workspace_change(change: WorkspaceChange) -> str:
    lines = [f"[cpn-renorm] configuration changes for {change.run_dir}:"]
    if change.source_changed:
        lines.append(f"  source: {change.previous_source!r} -> {change.source!r}")
    for item in change.changes:
        old = json.dumps(item["old"], ensure_ascii=False, default=str)
        new = json.dumps(item["new"], ensure_ascii=False, default=str)
        lines.append(f"  {item['path']}: {old} -> {new}")
    lines.append("[cpn-renorm] stage impact:")
    lines.extend(f"  {stage}: {action}" for stage, action in change.actions.items())
    return "\n".join(lines)


def write_workspace_state(change: WorkspaceChange, *, command: str,
                          status: str) -> None:
    change.run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(change.run_dir / "run_state.json", {
        "schema": STATE_SCHEMA,
        "source": change.source,
        "config": change.config,
        "config_fingerprint": change.config_fingerprint,
        "command": command,
        "status": status,
    })
    atomic_copy(change.source, change.run_dir / "input.toml")

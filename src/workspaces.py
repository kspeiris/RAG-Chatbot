from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.config import Settings
from src.utils import ensure_dir


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "workspace"


class WorkspaceManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        ensure_dir(self.settings.root_data_dir)
        ensure_dir(self.settings.root_data_dir / "workspaces")
        ensure_dir(self.settings.root_qdrant_path / "workspaces")
        if not self.registry_path.exists():
            self._write_registry({"workspaces": []})

    @property
    def registry_path(self) -> Path:
        return self.settings.root_data_dir / "workspaces_registry.json"

    def list_workspaces(self) -> list[dict[str, Any]]:
        items = list(self._read_registry().get("workspaces", []))
        items.sort(key=lambda item: (str(item.get("last_used_at", "")), str(item.get("created_at", ""))), reverse=True)
        return items

    def get_workspace(self, workspace_id: str) -> dict[str, Any] | None:
        for workspace in self.list_workspaces():
            if str(workspace.get("id", "")) == workspace_id:
                return workspace
        return None

    def ensure_default_workspace(self) -> dict[str, Any]:
        workspaces = self.list_workspaces()
        if workspaces:
            return workspaces[0]
        return self.create_workspace("Default Workspace")

    def create_workspace(self, name: str) -> dict[str, Any]:
        clean_name = name.strip() or "Untitled Workspace"
        registry = self._read_registry()
        existing_ids = {str(item.get("id", "")) for item in registry.get("workspaces", [])}
        base_id = _slugify(clean_name)
        workspace_id = base_id
        while workspace_id in existing_ids:
            workspace_id = f"{base_id}-{uuid4().hex[:6]}"

        now = _utc_now()
        workspace = {
            "id": workspace_id,
            "name": clean_name,
            "created_at": now,
            "last_used_at": now,
        }
        registry.setdefault("workspaces", []).append(workspace)
        self._write_registry(registry)
        return workspace

    def touch_workspace(self, workspace_id: str) -> None:
        registry = self._read_registry()
        changed = False
        for workspace in registry.get("workspaces", []):
            if str(workspace.get("id", "")) == workspace_id:
                workspace["last_used_at"] = _utc_now()
                changed = True
                break
        if changed:
            self._write_registry(registry)

    def _read_registry(self) -> dict[str, Any]:
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _write_registry(self, payload: dict[str, Any]) -> None:
        self.registry_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

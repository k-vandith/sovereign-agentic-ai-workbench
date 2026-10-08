"""Simple role-based access control for the workbench."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"


_LEVEL = {Role.VIEWER: 1, Role.OPERATOR: 2, Role.ADMIN: 3}

PERMISSIONS = {
    "query": Role.VIEWER,
    "use_tools": Role.VIEWER,
    "ingest": Role.OPERATOR,
    "view_audit": Role.ADMIN,
    "manage_roles": Role.ADMIN,
}


@dataclass
class Principal:
    name: str
    role: Role = Role.OPERATOR

    def can(self, action: str) -> bool:
        required = PERMISSIONS.get(action, Role.ADMIN)
        return _LEVEL[self.role] >= _LEVEL[required]


def require(principal: Principal, action: str) -> None:
    if not principal.can(action):
        raise PermissionError(
            f"Role '{principal.role.value}' cannot perform '{action}'. "
            f"Required: {PERMISSIONS.get(action, Role.ADMIN).value}"
        )

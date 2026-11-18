"""
Permission-Aware AI Tool Registry & Function-Calling Dispatcher.

Provides a decorator-driven registry (`@register_tool`) that binds Python functions
to OpenAI-compatible JSON schemas, role-based access rules (`staff_only`, `superuser_only`),
per-user tool preference filtering, and real-time Server-Sent Events (SSE) status labels.
"""

import logging
import threading
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Set

logger = logging.getLogger(__name__)


class UserContext(Protocol):
    """Minimal user interface required for permission evaluation and context injection."""

    id: Any
    is_authenticated: bool
    is_staff: bool
    is_superuser: bool


# Optional pluggable callback `(user) -> Iterable[str]` returning tool names disabled by a user.
DisabledToolsProvider = Callable[[Any], Iterable[str]]

_call_count_lock = threading.Lock()
_tool_call_counts: Dict[str, int] = {}


def log_tool_call(tool_name: str, **kwargs: Any) -> int:
    """Thread-safely increments and logs the invocation count for a tool."""
    with _call_count_lock:
        _tool_call_counts[tool_name] = _tool_call_counts.get(tool_name, 0) + 1
        count = _tool_call_counts[tool_name]

    args_repr = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    logger.info("[intelligence] tool=%s call_count=%d args=(%s)", tool_name, count, args_repr)
    return count


def _default_django_disabled_tools_provider(user: Any) -> Set[str]:
    """
    Queries Django's `intelligence.UserToolPreference` model when running inside
    the full Django application; gracefully returns an empty set in standalone environments.
    """
    try:
        from django.apps import apps

        if not apps.ready:
            return set()
        user_pref_model = apps.get_model("intelligence", "UserToolPreference")
        return set(
            user_pref_model.objects.filter(user=user, is_enabled=False).values_list("tool_name", flat=True)
        )
    except Exception:
        return set()


class ToolRegistry:
    """
    Central registry managing AI agent tools, their JSON schemas, role permissions,
    and safe execution dispatching.
    """

    def __init__(self, disabled_tools_provider: Optional[DisabledToolsProvider] = None) -> None:
        self._schemas: List[Dict[str, Any]] = []
        self._map: Dict[str, Callable[..., Any]] = {}
        self._display_names: Dict[str, str] = {}
        self._descriptions: Dict[str, str] = {}
        self._categories: Dict[str, str] = {}
        self._staff_only: Dict[str, bool] = {}
        self._superuser_only: Dict[str, bool] = {}
        self._disabled_tools_provider = disabled_tools_provider or _default_django_disabled_tools_provider

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        display_name: Optional[str] = None,
        staff_only: bool = True,
        superuser_only: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator to register a tool function with its OpenAI function-calling JSON schema,
        live SSE display label, and role permissions (`staff_only=True` by default).
        """
        if not name or not isinstance(name, str):
            raise ValueError("Tool name must be a non-empty string.")

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            schema = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
            # Replace existing schema if re-registered
            self._schemas = [s for s in self._schemas if s["function"]["name"] != name]
            self._schemas.append(schema)
            self._map[name] = func
            self._display_names[name] = display_name or f"Running {name}..."
            self._descriptions[name] = description
            self._categories[name] = func.__module__.split(".")[-1]
            self._staff_only[name] = bool(staff_only)
            self._superuser_only[name] = bool(superuser_only)
            return func

        return decorator

    def is_tool_allowed_for_user(self, name: str, user: Optional[Any]) -> bool:
        """
        Evaluates whether `user` has role permissions to access the tool `name`.
        """
        if name not in self._map:
            return False
        if not user or not getattr(user, "is_authenticated", False):
            return False

        is_superuser = bool(getattr(user, "is_superuser", False))
        is_staff = bool(getattr(user, "is_staff", False)) or is_superuser

        if self._superuser_only.get(name, False) and not is_superuser:
            return False

        if self._staff_only.get(name, True) and not is_staff:
            return False

        return True

    def get_schemas(self) -> List[Dict[str, Any]]:
        return list(self._schemas)

    def get_map(self) -> Dict[str, Callable[..., Any]]:
        return dict(self._map)

    def get_display_names(self) -> Dict[str, str]:
        return dict(self._display_names)

    def get_descriptions(self) -> Dict[str, str]:
        return dict(self._descriptions)

    def get_categories(self) -> Dict[str, str]:
        return dict(self._categories)

    def get_active_tools_for_user(self, user: Optional[Any]) -> Dict[str, Callable[..., Any]]:
        """
        Returns the mapping of tools accessible to `user`, filtering out tools
        restricted by role permissions or disabled in the user's preferences.
        """
        if not user or not getattr(user, "is_authenticated", False):
            return {}

        allowed_map = {
            name: func for name, func in self._map.items() if self.is_tool_allowed_for_user(name, user)
        }

        try:
            disabled_names = set(self._disabled_tools_provider(user) or ())
            if disabled_names:
                return {name: func for name, func in allowed_map.items() if name not in disabled_names}
        except Exception as exc:
            logger.warning("Failed to evaluate user tool preferences: %s", exc)

        return allowed_map

    def get_schemas_for_user(self, user: Optional[Any]) -> List[Dict[str, Any]]:
        """
        Returns OpenAI function-calling tool schemas filtered to only active and permitted tools for `user`.
        """
        active_map = self.get_active_tools_for_user(user)
        return [s for s in self._schemas if s["function"]["name"] in active_map]

    def execute(
        self,
        name: str,
        kwargs: Optional[Dict[str, Any]] = None,
        user: Optional[Any] = None,
    ) -> Any:
        """
        Executes a registered tool by name after verifying user permissions and active status.
        Returns a structured error message string if the tool is unregistered, disabled, or raises an error.
        """
        if name not in self._map:
            return f"Error: Tool '{name}' is not registered."

        active_tools = self.get_active_tools_for_user(user)
        if name not in active_tools:
            return f"Error: Tool '{name}' is not permitted or is disabled."

        func = active_tools[name]
        call_kwargs = dict(kwargs or {})

        if user and hasattr(user, "id") and name == "create_blog_post" and not call_kwargs.get("author_id"):
            call_kwargs["author_id"] = user.id

        try:
            return func(**call_kwargs)
        except TypeError as exc:
            logger.warning("Invalid arguments for tool '%s': %s", name, exc)
            return f"Error: Invalid arguments for tool '{name}': {exc}"
        except Exception as exc:
            logger.exception("Unhandled error executing tool '%s'", name)
            return f"Error executing tool '{name}': {exc}"


registry = ToolRegistry()
register_tool = registry.register

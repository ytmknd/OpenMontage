"""Base tool class implementing the expanded ToolContract.

Every tool in OpenMontage inherits from BaseTool. This enforces a uniform
interface for discovery, execution, cost estimation, and health reporting.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import subprocess
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

# cost_tracker only imports lib.config_model, so importing it here at module
# scope does not create an import cycle with base_tool.
from tools.cost_tracker import ApprovalRequiredError, BudgetExceededError, CostTracker


def _load_dotenv() -> None:
    """Load .env into os.environ once at import time.

    This ensures API keys are available before any tool is instantiated,
    even when tools are imported directly without going through the registry.
    Only sets variables that are not already in the environment.

    Delegates to lib.env_loader.load_env(), the canonical .env parser.
    """
    from lib.env_loader import load_env

    load_env()


_load_dotenv()


class ToolTier(str, Enum):
    CORE = "core"
    VOICE = "voice"
    ENHANCE = "enhance"
    GENERATE = "generate"
    SOURCE = "source"
    ANALYZE = "analyze"
    PUBLISH = "publish"


class ToolStability(str, Enum):
    EXPERIMENTAL = "experimental"
    BETA = "beta"
    PRODUCTION = "production"


class ToolStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"


class ToolRuntime(str, Enum):
    """Where and how a tool executes."""
    LOCAL = "local"            # Runs entirely on-device, free, no network
    LOCAL_GPU = "local_gpu"    # Runs on-device but needs GPU (VRAM)
    API = "api"                # Calls an external API, requires API key, costs money
    HYBRID = "hybrid"          # Can run locally OR via API (e.g., image_selector)


class ExecutionMode(str, Enum):
    SYNC = "sync"
    ASYNC = "async"


class Determinism(str, Enum):
    DETERMINISTIC = "deterministic"
    SEEDED = "seeded"
    STOCHASTIC = "stochastic"


class ResumeSupport(str, Enum):
    NONE = "none"
    FROM_START = "from_start"
    FROM_CHECKPOINT = "from_checkpoint"


@dataclass
class ResourceProfile:
    """Hardware resource envelope for a tool."""
    cpu_cores: int = 1
    ram_mb: int = 512
    vram_mb: int = 0
    disk_mb: int = 100
    network_required: bool = False


@dataclass
class RetryPolicy:
    """Safe retry behavior for a tool."""
    max_retries: int = 0
    backoff_seconds: float = 1.0
    retryable_errors: list[str] = field(default_factory=list)


@dataclass
class ToolResult:
    """Standard result returned by tool execution."""
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    error: Optional[str] = None
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    seed: Optional[int] = None
    model: Optional[str] = None


class BaseTool(ABC):
    """Abstract base class for all OpenMontage tools."""

    # --- Identity (override in subclasses) ---
    name: str = ""
    version: str = "0.1.0"
    tier: ToolTier = ToolTier.CORE
    stability: ToolStability = ToolStability.EXPERIMENTAL
    execution_mode: ExecutionMode = ExecutionMode.SYNC
    determinism: Determinism = Determinism.DETERMINISTIC
    runtime: ToolRuntime = ToolRuntime.LOCAL

    # --- Dependencies ---
    # For API tools, add "env:ENVVAR_NAME" to signal required API keys
    dependencies: list[str] = []
    install_instructions: str = ""

    # --- Capabilities ---
    capability: str = "generic"
    provider: str = "openmontage"
    capabilities: list[str] = []
    input_schema: dict = {}
    output_schema: dict = {}
    artifact_schema: dict = {}
    progress_schema: Optional[dict] = None
    supports: dict[str, Any] = {}
    best_for: list[str] = []
    not_good_for: list[str] = []
    provider_matrix: dict[str, Any] = {}

    # --- Resource & retry ---
    resource_profile: ResourceProfile = ResourceProfile()
    retry_policy: RetryPolicy = RetryPolicy()

    # --- Resume & idempotency ---
    resume_support: ResumeSupport = ResumeSupport.NONE
    idempotency_key_fields: list[str] = []

    # --- Side effects & fallback ---
    side_effects: list[str] = []
    fallback: Optional[str] = None
    fallback_tools: list[str] = []

    # --- Agent skills (Layer 3 references) ---
    # Names of installed agent skills in .agents/skills/ that teach the
    # underlying technology. The orchestrator uses these to load relevant
    # API knowledge when planning tool usage.
    agent_skills: list[str] = []

    # --- Verification ---
    user_visible_verification: list[str] = []

    # --- Optional telemetry / quality hints for the scoring engine ---
    # If set (0.0-1.0), lib/scoring.py uses these directly instead of falling
    # back to stability-based heuristics. Leave unset unless the tool has a
    # real measured or well-calibrated value.
    quality_score: Optional[float] = None
    historical_success_rate: Optional[float] = None
    latency_p50_seconds: Optional[float] = None

    # ---- Status reporting ----

    def get_status(self) -> ToolStatus:
        """Check if this tool's dependencies are satisfied."""
        try:
            self.check_dependencies()
            return ToolStatus.AVAILABLE
        except DependencyError:
            return ToolStatus.UNAVAILABLE

    def get_status_cached(self, ttl_seconds: float = 60.0) -> ToolStatus:
        """get_status() with a per-instance TTL cache. Env/config changes within
        the TTL window are not seen — call get_status() directly when freshness
        matters more than speed.
        """
        cached = getattr(self, "_status_cache", None)
        now = time.monotonic()
        if cached is not None:
            status, cached_at = cached
            if now - cached_at < ttl_seconds:
                return status
        status = self.get_status()
        self._status_cache = (status, now)
        return status

    def check_dependencies(self) -> None:
        """Verify all dependencies are installed. Raises DependencyError if not."""
        for dep in self.dependencies:
            if dep.startswith("cmd:"):
                cmd_name = dep[4:]
                if shutil.which(cmd_name) is None:
                    raise DependencyError(
                        f"Command {cmd_name!r} not found. {self.install_instructions}"
                    )
            elif dep.startswith("env:"):
                env_name = dep[4:]
                if not os.environ.get(env_name):
                    raise DependencyError(
                        f"Environment variable {env_name!r} not set. {self.install_instructions}"
                    )
            elif dep.startswith("python:"):
                module_name = dep[7:]
                try:
                    __import__(module_name)
                except ImportError:
                    raise DependencyError(
                        f"Python module {module_name!r} not installed. {self.install_instructions}"
                    )

    def get_info(self) -> dict[str, Any]:
        """Return full tool contract info for registry/discovery."""
        usage_location = inspect.getfile(self.__class__)
        return {
            "name": self.name,
            "version": self.version,
            "tier": self.tier.value,
            "capability": self.capability,
            "provider": self.provider,
            "stability": self.stability.value,
            "status": self.get_status().value,
            "execution_mode": self.execution_mode.value,
            "determinism": self.determinism.value,
            "runtime": self.runtime.value,
            "module_path": self.__class__.__module__,
            "usage_location": usage_location,
            "dependencies": self.dependencies,
            "install_instructions": self.install_instructions,
            "capabilities": self.capabilities,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "artifact_schema": self.artifact_schema,
            "supports": self.supports,
            "best_for": self.best_for,
            "not_good_for": self.not_good_for,
            "provider_matrix": self.provider_matrix,
            "resource_profile": {
                "cpu_cores": self.resource_profile.cpu_cores,
                "ram_mb": self.resource_profile.ram_mb,
                "vram_mb": self.resource_profile.vram_mb,
                "disk_mb": self.resource_profile.disk_mb,
                "network_required": self.resource_profile.network_required,
            },
            "resume_support": self.resume_support.value,
            "side_effects": self.side_effects,
            "fallback": self.fallback,
            "fallback_tools": self.fallback_tools or ([self.fallback] if self.fallback else []),
            "agent_skills": self.agent_skills,
            "related_skills": self.agent_skills,
            "user_visible_verification": self.user_visible_verification,
            "quality_score": self.quality_score,
            "historical_success_rate": self.historical_success_rate,
            "latency_p50_seconds": self.latency_p50_seconds,
        }

    # ---- Cost estimation ----

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        """Estimate cost in USD for the given inputs. Override for paid tools."""
        return 0.0

    def estimate_runtime(self, inputs: dict[str, Any]) -> float:
        """Estimate runtime in seconds. Override for long-running tools."""
        return 0.0

    # ---- Idempotency ----

    def idempotency_key(self, inputs: dict[str, Any]) -> str:
        """Compute a cache key from idempotency fields."""
        key_data = {k: inputs.get(k) for k in self.idempotency_key_fields}
        raw = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ---- Execution ----

    @abstractmethod
    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        """Run the tool. Subclasses must implement this."""
        ...

    def execute_tracked(
        self,
        inputs: dict[str, Any],
        tracker: CostTracker,
        *,
        operation: Optional[str] = None,
        approved: bool = False,
    ) -> ToolResult:
        """Run execute() under budget governance: estimate -> reserve -> execute
        (with retry per retry_policy) -> reconcile. Paid tools should be invoked
        through this wrapper so cost_log.json reflects reality.
        """
        estimated = self.estimate_cost(inputs)
        entry_id = tracker.estimate(
            self.name, operation or str(inputs.get("operation", "execute")), estimated
        )

        try:
            tracker.reserve(entry_id, approved=approved)
        except ApprovalRequiredError as exc:
            return ToolResult(
                success=False,
                error=str(exc),
                data={
                    "blocked_by": "approval_required",
                    "cost_entry_id": entry_id,
                    "estimated_usd": estimated,
                    "tool": self.name,
                },
            )
        except BudgetExceededError as exc:
            return ToolResult(
                success=False,
                error=str(exc),
                data={
                    "blocked_by": "budget_exceeded",
                    "cost_entry_id": entry_id,
                    "estimated_usd": estimated,
                    "tool": self.name,
                },
            )

        attempts_allowed = 1 + max(0, self.retry_policy.max_retries)
        result: Optional[ToolResult] = None
        attempts = 0
        for attempt_index in range(attempts_allowed):
            attempts += 1
            try:
                result = self.execute(inputs)
            except Exception as exc:  # noqa: BLE001 - convert any tool failure
                result = ToolResult(success=False, error=str(exc))

            if result.success:
                break

            attempts_remaining = attempts_allowed - attempts
            if attempts_remaining > 0 and self._is_retryable_error(result.error):
                time.sleep(self.retry_policy.backoff_seconds * (2 ** attempt_index))
                continue
            break

        tracker.reconcile(entry_id, result.cost_usd or 0.0, success=result.success)

        if result.data is None:
            result.data = {}
        result.data["cost_entry_id"] = entry_id
        result.data["estimated_usd"] = estimated
        result.data["attempts"] = attempts
        result.data["cost_tracked"] = True
        return result

    def _is_retryable_error(self, error: Optional[str]) -> bool:
        """Whether ``error`` matches one of this tool's declared retryable tokens."""
        if not error or not self.retry_policy.retryable_errors:
            return False
        lowered = error.lower()
        for token in self.retry_policy.retryable_errors:
            token_lower = token.lower()
            if token_lower == "rate_limit":
                if "rate limit" in lowered or "rate_limit" in lowered or "429" in lowered:
                    return True
            elif token_lower == "timeout":
                if "timeout" in lowered or "timed out" in lowered:
                    return True
            elif token_lower in lowered:
                return True
        return False

    def dry_run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Preflight check without side effects. Override for paid/publishing tools."""
        return {
            "tool": self.name,
            "estimated_cost_usd": self.estimate_cost(inputs),
            "estimated_runtime_seconds": self.estimate_runtime(inputs),
            "status": self.get_status().value,
            "would_execute": True,
        }

    # ---- CLI helper ----

    def run_command(
        self,
        cmd: list[str],
        *,
        timeout: Optional[int] = None,
        cwd: Optional[Path] = None,
    ) -> subprocess.CompletedProcess:
        """Run a subprocess command with standard error handling.

        On Windows, resolves .cmd/.bat wrappers (e.g. npx, npm) via
        shutil.which() so subprocess.run() can find them without shell=True.
        """
        resolved_cmd = list(cmd)
        if platform.system() == "Windows" and resolved_cmd:
            exe = shutil.which(resolved_cmd[0])
            if exe:
                resolved_cmd[0] = exe
        return subprocess.run(
            resolved_cmd,
            capture_output=True,
            text=True,
            # Force UTF-8 decoding. The default uses the OS locale (cp1252 on
            # Windows), which raises UnicodeDecodeError on a subprocess that
            # emits Unicode/emoji (e.g. Remotion's progress output), killing the
            # reader thread and potentially swallowing the real error text.
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=cwd,
            check=True,
        )


class DependencyError(Exception):
    """Raised when a tool's dependency is not satisfied."""
    pass

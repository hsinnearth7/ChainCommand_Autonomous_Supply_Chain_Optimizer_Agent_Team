"""Base agent class for all supply chain agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..data.schemas import AgentAction, SupplyChainEvent
from ..llm.base import BaseLLM
from ..tools.base_tool import BaseTool
from ..utils.logging_config import get_logger

log = get_logger(__name__)

MAX_ACTION_LOG = 1_000


class BaseAgent(ABC):
    """Abstract base class for all ChainCommand agents."""

    name: str = "base_agent"
    role: str = "Base agent"
    layer: str = "operational"  # strategic | tactical | operational | orchestration
    state_key: str = ""  # v2.0: key into SupplyChainState for state isolation

    def __init__(self, llm: BaseLLM, tools: List[BaseTool] | None = None) -> None:
        self.llm = llm
        self.tools: List[BaseTool] = tools or []
        self._action_log: List[AgentAction] = []
        self._active = True
        self._last_run: Optional[datetime] = None
        self._cycle_count = 0

    async def think(self, context: Dict[str, Any]) -> str:
        """Use LLM to reason about current context and decide next action."""
        tools_desc = "\n".join(
            f"- {t.name}: {t.description}" for t in self.tools
        )
        system_prompt = (
            f"You are {self.name}, a supply chain AI agent.\n"
            f"Role: {self.role}\n"
            f"Available tools: {tools_desc}\n"
            "Analyze the context and recommend actions."
        )
        prompt = f"Current context:\n{self._format_context(context)}\n\nWhat should we do?"
        response = await self.llm.generate(prompt, system=system_prompt)
        log.info("agent_think", agent=self.name, response_len=len(response))
        return response

    async def act(self, action: AgentAction) -> Dict[str, Any]:
        """Execute an action using the appropriate tool."""
        tool = self._find_tool(action.action_type)
        if tool is None:
            action.success = False
            action.error = f"Tool not found: {action.action_type}"
            self._action_log.append(action)
            return {"error": action.error}

        try:
            result = await tool.execute(**action.input_data)
            action.output_data = result
            action.success = True
        except Exception as exc:
            action.success = False
            action.error = str(exc)
            result = {"error": str(exc)}
            log.error("agent_act_error", agent=self.name, action=action.action_type, error=str(exc))

        self._action_log.append(action)
        if len(self._action_log) > MAX_ACTION_LOG:
            self._action_log = self._action_log[-MAX_ACTION_LOG:]
        return result

    @abstractmethod
    async def handle_event(self, event: SupplyChainEvent) -> None:
        """React to a supply chain event."""

    @abstractmethod
    async def run_cycle(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one full decision cycle."""

    def get_status(self) -> Dict[str, Any]:
        """Return agent status summary."""
        return {
            "name": self.name,
            "role": self.role,
            "layer": self.layer,
            "active": self._active,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "cycle_count": self._cycle_count,
            "actions_taken": len(self._action_log),
            "tools": [t.name for t in self.tools],
        }

    def _find_tool(self, name: str) -> Optional[BaseTool]:
        """Find a tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context dict into a readable string for LLM."""
        lines = []
        for key, value in context.items():
            if isinstance(value, list) and len(value) > 5:
                lines.append(f"{key}: [{len(value)} items]")
            elif isinstance(value, dict):
                lines.append(f"{key}: {value}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    @property
    def recent_actions(self) -> List[AgentAction]:
        return self._action_log[-20:]

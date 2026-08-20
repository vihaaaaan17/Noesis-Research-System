"""
tools/base_tool.py
---------------------------------------------------------------------
The BaseTool class and ToolResult contract that every tool inherits from.
---------------------------------------------------------------------
"""

from abc import ABC, abstractmethod
from typing import Optional, Any


class ToolResult:
    """
    Structured result contract for tool executions.
    Distinguishes successful tool outputs from tool failures.
    """

    def __init__(self, success: bool, output: Optional[str] = None, error: Optional[str] = None):
        self.success = success
        self.output = output
        self.error = error

    def __str__(self) -> str:
        if not self.success:
            return f"[Tool Error]: {self.error or 'Tool execution failed'}"
        return str(self.output or "")

    def __repr__(self) -> str:
        return f"ToolResult(success={self.success}, output={str(self.output)[:30]!r}, error={self.error!r})"


class BaseTool(ABC):
    """
    Abstract base class for all tools.
    Every tool must inherit from BaseTool, specify name and description, and implement run().
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, input_text: str) -> Any:
        """
        Execute the tool with the given input.
        Returns either a string or a ToolResult object.
        """
        pass

    def __repr__(self):
        return f"Tool(name={self.name!r})"

    def to_dict(self) -> dict:
        """Return a dict representation of the tool for system prompts."""
        return {
            "name": self.name,
            "description": self.description,
        }
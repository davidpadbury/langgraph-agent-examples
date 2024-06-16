from langchain_core.tools import BaseTool
from langchain_core.messages.tool import ToolCall, ToolMessage


def find_tool(name: str, tools: list[BaseTool]) -> BaseTool:
    """Look up a set of tools by name"""
    tool = next(filter(lambda tool: tool.name == name, tools), None)

    if not tool:
        raise ValueError(f"No tool found with name: {name}")

    return tool


def call_tool(tool_call: ToolCall, tools: list[BaseTool]) -> ToolMessage:
    tool = find_tool(tool_call["name"], tools)
    result = tool.invoke(tool_call["args"])

    return ToolMessage(tool_call_id=tool_call["id"], content=result)

from agent.tools import MemorySearchTool, ToolsmakerTool
from agent_build.agent1.tools import MemorySearchTool as LegacyMemorySearchTool
from agent_build.agent1.tools import ToolsmakerTool as LegacyToolsmakerTool


def test_agent1_tools_module_reexports_from_agent_tools():
    assert LegacyMemorySearchTool is MemorySearchTool
    assert LegacyToolsmakerTool is ToolsmakerTool

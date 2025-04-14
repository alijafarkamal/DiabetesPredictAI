from .agent import AgentPro
from typing import Any
from .tools import AresInternetTool, CodeEngine, YouTubeSearchTool, SlideGenerationTool # add more tools when available
ares_tool = AresInternetTool()
# code_tool = CodeEngine()
youtube_tool = YouTubeSearchTool()
# slide_tool = SlideGenerationTool()
__all__ = ['AgentPro', 'ares_tool', 'youtube_tool']
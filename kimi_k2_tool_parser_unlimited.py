from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.tool_parsers.kimi_k2_tool_parser import KimiK2ToolParser


@ToolParserManager.register_module("kimi_k2_unlimited")
class KimiK2UnlimitedToolParser(KimiK2ToolParser):
    """KimiK2 tool parser with buffer/section limits raised to 10 MB
    so they are never triggered during normal inference."""

    def __init__(self, tokenizer, tools=None):
        super().__init__(tokenizer, tools)
        self.buffer_max_size = 10 * 1024 * 1024
        self.max_section_chars = 10 * 1024 * 1024

from dataclasses import dataclass, field

@dataclass
class UIState:
    chat_history: list = field(default_factory=list)
    context: str = ""
    system_prompt: str = ""
    token_usage: int = 0
    playlist: list = field(default_factory=list)

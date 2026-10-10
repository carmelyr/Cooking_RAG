from project import config

class Conversation:
    def __init__(self, tokenizer, system_prompt: str, context: str, history: list):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.context = context
        self.history = history

    def add_user(self, text: str):
        self.history.append({"role": "user", "content": text})

    def add_assistant(self, text: str):
        self.history.append({"role": "assistant", "content": text})

    def build_prompt(self) -> str:
        prompt = ""
        prompt += "<|system|>\n" + self.system_prompt.strip() + "\n\n" # system instructions
        prompt += "<|context|>\n" # context block

        if self.context.strip():
            prompt += self.context.strip() + "\n\n" # Add retrieved context
        else:
            prompt += "[NO RETRIEVED CONTEXT]\n\n"

        for turn in self.history: # conversation history
            role = turn["role"]
            content = turn["content"].strip()
            if role == "user":
                prompt += "<|user|>\n" + content + "\n\n"
            elif role == "assistant":
                prompt += "<|assistant|>\n" + content + "\n\n"

        prompt += "<|assistant|>\n"

        return prompt

    def truncate(self): # Post-processing step
        while True:
            prompt = self.build_prompt()
            token_count = len(self.tokenizer.encode(prompt))

            if token_count <= config.MAX_CONTEXT_TOKENS:
                break

            if len(self.history) > 2: # If there are more then 2 turns, remove the oldest
                print("Truncating conversation history!!!!")
                self.history.pop(0)
            else:
                break

    def token_count(self) -> int:
        return len(self.tokenizer.encode(self.build_prompt()))

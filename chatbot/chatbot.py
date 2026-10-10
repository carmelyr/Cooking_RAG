
from project import config
from project.chatbot.prompt import Conversation
from project.chatbot.model import ChatModel

class Chatbot:
    def __init__(self):
        self.model = ChatModel("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    def chat(self, user_input: str, context, history: list):
        # Create a fresh Conversation for each call
        conversation = Conversation(
            tokenizer=self.model.tokenizer, # Tokenizer: BPE
            system_prompt=config.SYSTEM_PROMPT,
            context=context,
            history=history
        )

        conversation.add_user(user_input)
        conversation.truncate()

        prompt = conversation.build_prompt() # Final prompt with system instructions, retrieved context, and conversation history
        reply = self.model.generate_reply(prompt)

        # Add assistant's reply to the conversation history
        conversation.add_assistant(reply)

        # Calculate token usage
        prompt_tokens = len(self.model.tokenizer.encode(prompt))
        reply_tokens = len(self.model.tokenizer.encode(reply))
        used_tokens = prompt_tokens + reply_tokens # input + output tokens

        return reply, conversation.history, used_tokens

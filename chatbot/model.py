import torch
from project import config
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

class StopOnTokens(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings, start_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.start_len = start_len # Prompt length in tokens
    def __call__(self, input_id, _)-> bool:
        gen_ids = input_id[0, self.start_len:] # New tokens
        decoded = self.tokenizer.decode(gen_ids,skip_special_tokens=True) # Decode only the generated part
        return any(s in decoded for s in self.stop_strings)

class ChatModel:
    def __init__(self, model_name: str, device=None):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.dtype = torch.float16 if self.device.type in ["cuda", "mps"] else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=self.dtype,
        ).to(self.device)

        self.model.eval()

    def generate_reply(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1] # Get the length of the prompt in tokens

        stopper = StopOnTokens(
            tokenizer=self.tokenizer,
            stop_strings=config.STOP_TOKENS,
            start_len=prompt_len,
        )
        
        print("Start generating reply...")
        out = self.model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=config.USE_SAMPLING, # Less deterministic if True
            top_p=config.TOP_P,
            temperature=config.TEMPERATURE,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([stopper]),
        )
        print("Reply generation completed.")

        gen_ids = out[0][prompt_len:] # Slice off the prompt tokens to get only the generated part
        reply = self.tokenizer.decode(gen_ids, skip_special_tokens=True) # Decode the generated part of the output

        return self._postprocess(reply)

    def _postprocess(self, reply: str) -> str:
        cut = len(reply) # Initial cut is the full length of the generated reply
        for marker in config.STOP_TOKENS:
            idx = reply.find(marker)
            if idx !=-1 and idx < cut: # If the marker is found and is earlier than the current cut, update the cut position
                cut = idx
        reply = reply[:cut] # Cut off anything after the earliest stop token

        reply = reply.strip()
        if not reply:
            return "[Empty Reply]"
        return reply

from project import config
from project.chatbot.chatbot import Chatbot
from project.retrieval.recipe_retrieval import RecipeRetriever
from project.retrieval.video_retrieval import VideoRetriever

class RAGPipeline:
    def __init__(self):
        self.recipe_retriever = RecipeRetriever(
            title_index_path= config.FAISS_TITLE_INDEX_PATH,
            recipe_ids_path=config.RECIPE_IDS_PATH
        )
        self.video_retriever = VideoRetriever(
            clip_index_path= config.FAISS_CLIP_INDEX_PATH,
            clip_metadata_path= config.CLIP_METADATA_PATH
        )
        self.chatbot = Chatbot()
    
    def run_rag_pipeline(self, query: str, history: list):
        # Recipe extraction
        recipes = self.recipe_retriever.similarity_search(query, config.TOP_K_TITLE)
        context = self.recipe_retriever.format_as_context(recipes)

        # Chatbot response generation
        generated_response, updated_history, used_tokens = self.chatbot.chat(query, context, history)

        # Instructional Clip extraction
        instruction_clips = []
        recipe_instructions = self.recipe_retriever.get_recipe_instructions(generated_response, recipes) # Try to extract instructions from chatbot response
        for instructions in recipe_instructions:
            clips = self.video_retriever.similarity_search(instructions['instruction'], config.TOP_K_VIDEO_CLIPS)
            instruction_clips.append({
                "step_id": instructions['step_id'],
                "instruction": instructions['instruction'],
                "clips": clips # list of dicts with clip metadata
            })
        return instruction_clips, context, updated_history, used_tokens

if __name__ == "__main__":
    pipeline = RAGPipeline()
    history = [] # initialize empty history

    # For testing
    while True:
        user = input("User: ")
        instruction_clips, context, history, used_tokens = pipeline.run_rag_pipeline(user, history)
        print("Assistant:", history[-1]['content'])
        print("Token usage:", used_tokens)
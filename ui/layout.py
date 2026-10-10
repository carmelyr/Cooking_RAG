import gradio as gr
from project.ui.state import UIState
from project import config
from project.ui.utils import playlist_to_df
from project.ui.handlers import create_submit_handler, on_playlist_select

def build_ui(pipeline):
    with gr.Blocks(title="Chat Interaction UI", theme=gr.themes.Ocean()) as demo:

        state = gr.State(UIState()) # Store page state in a Gradio State object
        
        on_submit = create_submit_handler(pipeline)

        with gr.Row():

            with gr.Column(scale=3):

                with gr.Row():
                    gr.Markdown("## CookBook AI") # Title of page


                with gr.Column(): # System prompt and context display
                    gr.Textbox(
                        label="System Prompt",
                        value=config.SYSTEM_PROMPT,
                        interactive=False,
                        lines=3,
                    )

                    context_box = gr.Textbox( # Context display box
                        label="Injected Context",
                        value="No context loaded.",
                        lines=5,
                        interactive=False,
                    )

                chatbot = gr.Chatbot( # Main chat interface where conversation history is displayed
                    value=[
                        {"role": "assistant", "content": "Welcome. How can I assist you today?"}
                    ],
                    height=500,
                    type='messages',
                )

                with gr.Row(): # User input and send button
                    user_input = gr.Textbox(
                        placeholder="Type your message here...",
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("Send", scale=1)

            with gr.Column(scale=2): # Right panel for video exploration

                with gr.Column(elem_classes="panel"):
                    with gr.Row():
                        gr.Markdown("### 🎬 Video Exploration")
                        token_display = gr.Markdown("**Total Token Usage:** 0")

                with gr.Column():
                    
                    with gr.Column(scale=1):
                        video_player = gr.Video(
                            label="Video Player",
                            autoplay=True,
                        )

                        gr.Markdown("##### Video Playlist")

                        playlist = gr.Dataframe(
                            value=playlist_to_df(),
                            headers=["Step", "Instruction", "Start (s)", "Score"],
                            interactive=False,
                            row_count=(5, "dynamic"),
                        )

        send_btn.click( # When user clicks send button, run the RAG pipeline and update all relevant UI components with the new state
            on_submit,
            inputs=[user_input, state],
            outputs=[chatbot, user_input, context_box, token_display, playlist, state]
        )

        user_input.submit(
            on_submit,
            inputs=[user_input, state],
            outputs=[chatbot, user_input, context_box, token_display, playlist, state]
        )
        playlist.select( # When user selects a row in the playlist, update the video player to show the corresponding clip
            on_playlist_select,
            inputs=[state],
            outputs=[video_player]
        )

    return demo
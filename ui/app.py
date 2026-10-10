import gradio as gr
from project import config
from project.pipeline import RAGPipeline
from project.ui.layout import build_ui

def main():
    pipeline = RAGPipeline()
    app = build_ui(pipeline)
    app.queue(default_concurrency_limit=1)
    app.launch(allowed_paths=[str(config.VIDEO_ROOT.resolve())])

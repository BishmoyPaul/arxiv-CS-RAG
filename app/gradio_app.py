from __future__ import annotations

from app.dependencies import get_config, get_generation_service, get_pipeline
from app.gradio_controller import APP_HEADER_TEXT, SEARCH_RESULTS_HEADER, GradioController


def build_controller() -> GradioController:
    return GradioController(
        config=get_config(),
        pipeline=get_pipeline(),
        generation_service=get_generation_service(),
    )


def build_demo(controller: GradioController | None = None):
    import gradio as gr

    active_controller = controller or build_controller()

    def update_with_rag_md(message: str, llm_results_use: int, database_choice: str):
        state = active_controller.prepare_search(message, llm_results_use, database_choice)
        if state.error_message:
            gr.Warning(state.error_message)
        if state.warning_message:
            gr.Warning(state.warning_message)
        return state.search_results_markdown, state.prompt

    def ask_llm(prompt: str, llm_model_picked: str, stream_outputs: bool):
        yield from active_controller.generate_answer(prompt, llm_model_picked, stream_outputs)

    with gr.Blocks(title="ArXiv CS RAG", theme=gr.themes.Soft()) as demo:
        gr.Markdown(APP_HEADER_TEXT)

        with gr.Group():
            msg = gr.Textbox(label="Search", placeholder="e.g., What is Mixtral?")
            with gr.Accordion("Advanced Settings", open=False):
                llm_model = gr.Dropdown(
                    choices=list(active_controller.llm_models_to_choose),
                    value=active_controller.default_llm_model,
                    label="LLM Model",
                )
                llm_results = gr.Slider(5, 20, value=10, step=1, label="Top n results as context")
                database_src = gr.Dropdown(
                    choices=active_controller.database_choices,
                    value=active_controller.default_database_choice,
                    label="Search Source",
                )
                stream_results = gr.Checkbox(value=True, label="Stream output")

        output_text = gr.Textbox(
            label="LLM Answer",
            placeholder="The model's answer will appear here...",
            interactive=False,
            lines=8,
        )
        input_prompt = gr.Textbox(visible=False)
        gr_md = gr.Markdown(SEARCH_RESULTS_HEADER)

        msg.submit(
            fn=update_with_rag_md,
            inputs=[msg, llm_results, database_src],
            outputs=[gr_md, input_prompt],
        ).then(
            fn=ask_llm,
            inputs=[input_prompt, llm_model, stream_results],
            outputs=[output_text],
        )

    return demo


def main() -> None:
    build_demo().queue().launch()


if __name__ == "__main__":
    main()

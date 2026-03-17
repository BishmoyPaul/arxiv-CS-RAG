from __future__ import annotations

from collections.abc import Iterator

from .config import AppConfig


DEFAULT_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
]


class GeminiGenerationService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def generate_answer(self, prompt: str, model_name: str) -> str:
        if not prompt or not prompt.strip():
            return "Error: The generated prompt is empty. Please try a different query."

        if model_name == "None":
            return "LLM Model is disabled."

        if not self.config.gemini_api_key:
            return "Error: GEMINI_API_KEY is not configured. Cannot contact the LLM."

        try:
            response = self._generate_content(prompt, model_name, stream=False)
            return self._extract_non_stream_text(response)
        except Exception as exc:
            return f"An error occurred with the Gemini API: {exc}"

    def stream_answer(self, prompt: str, model_name: str) -> Iterator[str]:
        if not prompt or not prompt.strip():
            yield "Error: The generated prompt is empty. Please try a different query."
            return

        if model_name == "None":
            yield "LLM Model is disabled."
            return

        if not self.config.gemini_api_key:
            yield "Error: GEMINI_API_KEY is not configured. Cannot contact the LLM."
            return

        try:
            response = self._generate_content(prompt, model_name, stream=True)
            output = ""
            for chunk in response:
                try:
                    text = chunk.parts[0].text
                except (AttributeError, IndexError):
                    continue
                output += text
                yield output
            if not output:
                yield "Model returned an empty or blocked stream. This may be due to the safety settings or the nature of the prompt."
        except Exception as exc:
            yield f"An error occurred with the Gemini API: {exc}"

    def _generate_content(self, prompt: str, model_name: str, stream: bool):
        import google.generativeai as genai

        genai.configure(api_key=self.config.gemini_api_key)
        model = genai.GenerativeModel(model_name.split("/")[-1])
        return model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.config.generation.temperature,
                max_output_tokens=self.config.generation.max_output_tokens,
                top_p=self.config.generation.top_p,
            ),
            stream=stream,
            safety_settings=DEFAULT_SAFETY_SETTINGS,
        )

    @staticmethod
    def _extract_non_stream_text(response: object) -> str:
        try:
            return response.parts[0].text
        except (AttributeError, IndexError):
            return "Model returned an empty or blocked response."

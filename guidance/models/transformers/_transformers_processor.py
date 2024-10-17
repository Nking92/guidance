from dataclasses import dataclass
import inspect
import io
import logging
from transformers import AutoProcessor
from typing import List, Optional
from PIL import Image

from guidance.models._model import Modality, PromptMedia
from guidance.models.transformers._transformers_tokenizer import TransformersTokenizer


logger = logging.getLogger(__name__)


@dataclass
class TransformersInputProcessorResult:
    model_inputs: dict
    token_ids: list[int]
    # If -1, assume final media token can't be determined
    last_media_token_index: int = -1


class TransformersInputProcessor:
    def load_processor(self, model_name: str) -> TransformersTokenizer:
        """ Load the processor and tokenizer for the model, then return the tokenizer. """
        pass

    def process_inputs(self, prompt: str, media: List[PromptMedia]) -> TransformersInputProcessorResult:
        """ Process inputs for the model using prompt and media. Return a dictionary of inputs. """
        pass


class Phi3VisionInputProcessor(TransformersInputProcessor):
    def load_processor(self, model_name: str) -> TransformersTokenizer:
        # Processor handles tokenization and image processing
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        # Hack: the sp_whitespace=True argument is necessary because the Phi 3 Vision tokenizer
        # uses sentencepiece whitespace conventions but does not have an sp_model attribute
        tokenizer = TransformersTokenizer(model_name, self.processor.tokenizer, sp_whitespace=True)
        return tokenizer


    def process_inputs(self, prompt: str, media: List[PromptMedia]) -> TransformersInputProcessorResult:
        # TODO: See if media bytes are copied in memory here, resulting in high memory usage
        # Map Guidance placeholders to Phi 3 Vision format and make list of images for processing
        images = []
        processed_prompt = prompt
        image_counter = 1
        for m in media:
            if m.modality != Modality.IMAGE:
                raise ValueError(f"Unsupported non-image modality: {m.modality}")
            processed_prompt = processed_prompt.replace(
                m.prompt_placeholder, f"<|image_{image_counter}|>"
            )
            images.append(Image.open(io.BytesIO(m.data)))
            image_counter += 1
        logger.debug("Transformed prompt: %s -> ", prompt, processed_prompt)

        model_inputs = self.processor(
            text=processed_prompt,
            images=images if len(images) > 0 else None,
            return_tensors="pt",
        )

        tokens = model_inputs["input_ids"][0].tolist()

        # Find the last multimodal (negative) token in the sequence, if any
        # Note: Phi 3 vision uses a convention of negative tokens for multimodal inputs
        # Do not assume other models will use this convention
        last_image_token_index = -1
        for i, token in enumerate(reversed(tokens)):
            if token < 0:
                last_image_token_index = len(tokens) - i - 1
                break
        return TransformersInputProcessorResult(
            model_inputs=model_inputs,
            token_ids=tokens,
            last_media_token_index=last_image_token_index
        )


def create_input_processor(model=None, input_processor=None) -> Optional[TransformersInputProcessor]:
    """Finds the best input processor.

    Order of precedence:
    - If input_processor is a TransformersInputProcessor, use it directly
    - If model is string, attempt to load a supported input processor for the model
    - Else return None
    """
    if isinstance(input_processor, TransformersInputProcessor):
        if type(input_processor) is TransformersInputProcessor:
            raise Exception(
                "You can't use the base TransformersInputProcessor class directly. Create or use a subclass instead."
            )
        return input_processor

    if model == "microsoft/Phi-3-vision-128k-instruct":
        return Phi3VisionInputProcessor()
    else:
        return None


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
    @property
    def tokenizer(self) -> TransformersTokenizer:
        raise NotImplementedError

    def process(self, prompt: str, media: List[PromptMedia]) -> TransformersInputProcessorResult:
        """ Process inputs for the model using prompt and media. Return a dictionary of inputs. """
        raise NotImplementedError


class Phi3VisionInputProcessor(TransformersInputProcessor):
    def __init__(self, model_name: str):
        # Processor handles tokenization and image processing
        self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        # Hack: the sp_whitespace=True argument is necessary because the Phi 3 Vision tokenizer
        # uses sentencepiece whitespace conventions but does not have an sp_model attribute
        self._tokenizer = TransformersTokenizer(model_name, self._processor.tokenizer, sp_whitespace=True)

    @property
    def tokenizer(self) -> TransformersTokenizer:
        return self._tokenizer

    def process(self, prompt: str, media: List[PromptMedia]) -> TransformersInputProcessorResult:
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

        model_inputs = self._processor(
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


class Llama3VisionInputProcessor(TransformersInputProcessor):
    def __init__(self, model_name: str):
        self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self._tokenizer = TransformersTokenizer(model_name, self._processor.tokenizer)

    @property
    def tokenizer(self) -> TransformersTokenizer:
        return self._tokenizer

    def process(self, prompt: str, media: List[PromptMedia]) -> TransformersInputProcessorResult:
        if len(media) > 1:
            raise ValueError("Llama3Vision only supports a single image input at the beginning of the prompt.")
        elif len(media) == 1:
            m = media[0]
            if m.modality != Modality.IMAGE:
                raise ValueError(f"Unsupported non-image modality: {m.modality}")
            processed_prompt = prompt.replace(
                m.prompt_placeholder, f"<|image|>"
            )
            image = Image.open(io.BytesIO(m.data))
            logger.debug("Transformed prompt: %s -> ", prompt, processed_prompt)
        else:
            processed_prompt = prompt
            image = None
            last_image_token_index = -1

        model_inputs = self._processor(
            image,
            processed_prompt,
            return_tensors="pt",
        )

        tokens = model_inputs["input_ids"][0].tolist()

        if image is not None:
            # This is just a constant value obtained by manually inspecting the tokenizer
            # using the <|image|> token
            special_image_token_id = 128256
            last_image_token_index = tokens.index(special_image_token_id)
        else:
            last_image_token_index = -1

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
        return Phi3VisionInputProcessor(model)
    elif model == "meta-llama/Llama-3.2-11B-Vision":
        return Llama3VisionInputProcessor(model)
    else:
        return None


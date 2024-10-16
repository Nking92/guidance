from dataclasses import dataclass
import io
import logging
from transformers import AutoProcessor
from typing import List
from PIL import Image

from guidance.models._model import Modality
from guidance.models.transformers._transformers_tokenizer import TransformersTokenizer


logger = logging.getLogger(__name__)


@dataclass
class TransformersInputProcessorResult:
    model_inputs: dict
    token_ids: list[int]
    # If -1, assume final media token can't be determined
    last_media_token_index: int = -1


@dataclass
class PromptMedia:
    prompt_placeholder: str
    modality: Modality
    data: bytes


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
        self.tokenizer = TransformersTokenizer(model_name, self.processor.tokenizer, sp_whitespace=True)
        return self.tokenizer


    def process_inputs(self, prompt: str, media: List[PromptMedia]) -> TransformersInputProcessorResult:
        # TODO: See if media bytes are copied in memory here, resulting in high memory usage
        # Map Guidance placeholders to Phi 3 Vision format and make list of images for processing
        images = []
        processed_prompt = prompt
        image_counter = 1
        for m in media:
            if m.modality != Modality.IMAGE.name:
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


def load_processor_class(chat_template=None):
    """Utility method to find the best chat template.

    Order of precedence:
    - If it's a chat template class, use it directly
    - If it's a string, check the cache of popular model templates
    - If it's a string and not in the cache, try to create a class dynamically
    - [TODO] If it's a string and can't be created, default to ChatML and raise a warning
    - If it's None, default to ChatML and raise a warning
    """
    if inspect.isclass(chat_template) and issubclass(chat_template, ChatTemplate):
        if chat_template is ChatTemplate:
            raise Exception(
                "You can't use the base ChatTemplate class directly. Create or use a subclass instead."
            )
        return chat_template

    elif isinstance(chat_template, str):
        # First check the cache of popular model types
        # TODO: Expand keys of cache to include aliases for popular model types (e.g. "llama2, phi3")
        # Can possibly accomplish this with an "aliases" dictionary that maps all aliases to the canonical key in cache
        if chat_template in CHAT_TEMPLATE_CACHE:
            return CHAT_TEMPLATE_CACHE[chat_template]
        # TODO: Add logic here to try to auto-create class dynamically via _template_class_from_string method

    # Only warn when a user provided a chat template that we couldn't load
    if chat_template is not None:
        warnings.warn(
            f"""Chat template {chat_template} was unable to be loaded directly into guidance.
                        Defaulting to the ChatML format which may not be optimal for the selected model. 
                        For best results, create and pass in a `guidance.ChatTemplate` subclass for your model."""
        )

    # By default, use the ChatML Template. Warnings to user will happen downstream only if they use chat roles.
    return ChatMLTemplate

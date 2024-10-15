import logging
import io
import os
from typing import List, Optional

try:
    import torch
except ModuleNotFoundError:
    pass

import llguidance
from transformers import AutoModelForCausalLM, AutoProcessor

from guidance._parser import TokenParser, serialize_grammar, process_prompt
from guidance.models._model import (
    Engine,
    Model,
    modality_pattern,
    Modality
)
from guidance.models.transformers._transformers import PromptMedia, TransformersEngine, TransformersInputProcessor, TransformersInputProcessorResult, TransformersTokenizer

try:
    from PIL import Image
    has_pillow = True
except ModuleNotFoundError:
    has_pillow = False

logger = logging.getLogger(__name__)


class TransformersPhi3VisionEngine(TransformersEngine):
    def __init__(
        self,
        model="microsoft/Phi-3-vision-128k-instruct",
        compute_log_probs=False,
        **kwargs,
    ):
        if not has_pillow:
            raise Exception("Please install pillow with `pip install pillow` to use Phi 3 Vision")

        # Processor handles tokenization and image processing
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        # Hack: the sp_whitespace=True argument is necessary because the Phi 3 Vision tokenizer
        # uses sentencepiece whitespace conventions but does not have an sp_model attribute
        self.tokenizer = TransformersTokenizer(model, self.processor.tokenizer, sp_whitespace=True)
        super().__init__(self.model, self.tokenizer, compute_log_probs, **kwargs)


    def start(self, prompt: bytes, grammar, media: dict, ensure_bos_token=True) -> TokenParser:
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        elif isinstance(prompt, TokenParser):
            raise NotImplementedError(
                "Still need to implement support for extending a full Parser state."
            )
        elif not isinstance(prompt, str):
            raise Exception("The passed prompt is of an unknown type!")

        # Map Guidance placeholders to Phi 3 Vision format
        # and make list of images for processing
        images = []
        processed_prompt = prompt
        # TODO: This step can probably be hidden from input processor subclass
        matches = {}
        for match in modality_pattern.finditer(prompt):
            match_str = match.group(0)
            modality_type = match.group(1)
            if modality_type != Modality.IMAGE.name:
                logger.debug("Skipping non-image modality: %s", match_str)
                continue
            media_id = match.group(2)
            if match_str not in matches:
                matches[match_str] = media_id 

        image_counter = 1
        # TODO: we can transform matches and media dict into something easier to work with here
        # This work actually needs to happen in the subclass
        for match in matches.keys():
            processed_prompt = processed_prompt.replace(
                match, f"<|image_{image_counter}|>"
            )
            media_key = matches[match]
            images.append(Image.open(io.BytesIO(media[media_key])))
            image_counter += 1
        logger.debug("Transformed prompt: %s -> ", prompt, processed_prompt)

        model_inputs = self.processor(
            text=processed_prompt,
            images=images if len(images) > 0 else None,
            return_tensors="pt",
        ).to(self.device)

        # We will reuse everything except input_ids (attention_mask, pixel_values, image_sizes)
        self.model_inputs = model_inputs

        tokens = model_inputs["input_ids"][0].tolist()

        serialized_grammar = serialize_grammar(grammar)
        ll_tokenizer = llguidance.LLTokenizer(
            llguidance.TokenizerWrapper(self.tokenizer)
        )
        ll_interpreter = llguidance.LLInterpreter(
            ll_tokenizer,
            serialized_grammar,
            log_level=int(os.environ.get("LLGUIDANCE_LOG_LEVEL", "1")),
        )
        if ensure_bos_token and self.tokenizer.bos_token_id is not None:
            if self.tokenizer.bos_token_id is None:
                logger.warning("Tokenizer does not have a BOS token, but ensure_bos_token is True")
            bos_token_id = self.tokenizer.bos_token_id
        else:
            bos_token_id = None

        # Find the last multimodal (negative) token in the sequence, if any
        # Note: Phi 3 vision uses a convention of negative tokens for multimodal inputs
        # Do not assume other models will use this convention
        last_multimodal_index = -1
        for i, token in enumerate(reversed(tokens)):
            if token < 0:
                last_multimodal_index = len(tokens) - i - 1
                break

        # Process tokens and grammar state machine beginning from the last multimodal token
        if last_multimodal_index != -1:
            processed_tokens = process_prompt(tokens[last_multimodal_index+1:], ll_interpreter, bos_token_id)
            prompt_tokens = tokens[:last_multimodal_index+1] + processed_tokens
        else:
            prompt_tokens = process_prompt(tokens, ll_interpreter, bos_token_id)

        return TokenParser(ll_interpreter, prompt_tokens)


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
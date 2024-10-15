from dataclasses import dataclass
import logging
import os
import llguidance

from typing import List, Optional, Sequence, Union

from guidance._parser import TokenParser, process_prompt, serialize_grammar
from guidance.models.transformers._transformers_phi3v import TransformersPhi3VisionEngine
from guidance.models.transformers._transformers_tokenizer import TransformersTokenizer
from guidance.models._model import modality_pattern

try:
    import torch
except ModuleNotFoundError:
    pass


try:
    import transformers as transformers_package

    has_transformers = True
except ModuleNotFoundError:
    has_transformers = False


from .._model import Engine, Modality, Model

# Formed by comparing model and tokenizer from_pretrained methods
# transformers/models/auto/auto_factory.py
# transformers/models/auto/tokenization_auto.py
_COMMON_TRANSFORMERS_KWARGS = [
    "cache_dir",
    "force_download",
    "proxies",
    "resume_download",
    "revision",
    "subfolder",
    "trust_remote_code",
]

logger = logging.getLogger(__name__)

def load_transformers_model(model, **kwargs):
    # intantiate the model if needed
    if isinstance(model, str):

        # make sure transformers is installed
        if not has_transformers:
            raise Exception(
                "Please install transformers with `pip install transformers` in order to use guidance.models.Transformers!"
            )
        model = transformers_package.AutoModelForCausalLM.from_pretrained(model, **kwargs)
    return model


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


class TransformersEngine(Engine):
    def __init__(self, model, tokenizer, compute_log_probs: bool, chat_template=None, input_processor: Optional[TransformersInputProcessor]=None, **kwargs):
        # fill in default model value
        if model is None:
            model = os.environ.get("TRANSFORMERS_MODEL", None)
        if model is None:
            try:
                with open(os.path.expanduser("~/.transformers_model"), "r") as file:
                    model = file.read().replace("\n", "")
            except:
                pass

        self.model_obj = load_transformers_model(model, **kwargs)

        if not isinstance(model, str):
            self.model = model.__class__.__name__
        self.device = self.model_obj.device  # otherwise note the current device

        self._past_key_values = None
        self._cached_logits = None
        self._cached_token_ids: list[int] = []

        # Set attr for malformed tokenizer hack.
        # If more models start doing this, generalize into a util function.
        if hasattr(self.model_obj.config, "model_type"):
            if self.model_obj.config.model_type in ["phi3"]:
                self._disable_retokenize_check = True

        # Automatically fill common args between Transformers
        # model and tokenizer
        passed_common_kwargs = {}
        for arg_name in _COMMON_TRANSFORMERS_KWARGS:
            if arg_name in kwargs:
                passed_common_kwargs[arg_name] = kwargs[arg_name]

        # Create the tokenizer
        if input_processor is not None:
            if tokenizer is not None:
                raise ValueError("Cannot specify both tokenizer and input_processor")
            self.input_processor = input_processor
            tokenizer = input_processor.load_processor()
        if tokenizer is TransformersTokenizer:
            my_tokenizer = tokenizer
        else:
            my_tokenizer = TransformersTokenizer(
                model, tokenizer, chat_template, **passed_common_kwargs
            )

        super().__init__(
            my_tokenizer,
            compute_log_probs=compute_log_probs,
        )


    def start(self, prompt: bytes, grammar, media: Optional[dict]=None, ensure_bos_token=True) -> TokenParser:
        if self.input_processor is None:
            return super().start(prompt, grammar, media, ensure_bos_token)

        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        elif isinstance(prompt, TokenParser):
            raise NotImplementedError(
                "Still need to implement support for extending a full Parser state."
            )
        elif not isinstance(prompt, str):
            raise Exception("The passed prompt is of an unknown type!")

        prompt_media: List[PromptMedia] = []
        for match in modality_pattern.finditer(prompt):
            match_str = match.group(0)
            modality_name = match.group(1)
            modality = Modality[modality_name]
            media_id = match.group(2)
            media_data = media[media_id]
            prompt_media.append(PromptMedia(match_str, modality, media_data))

        result = self.input_processor.process_inputs(prompt, prompt_media)
        self.model_inputs = result.model_inputs.to(self.device)
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

        tokens = result.token_ids
        if result.last_media_token_index != -1:
            processed_tokens = process_prompt(tokens[result.last_media_token_index+1:], ll_interpreter, bos_token_id)
            prompt_tokens = tokens[:result.last_media_token_index+1] + processed_tokens
        else:
            prompt_tokens = process_prompt(tokens, ll_interpreter, bos_token_id)

        return TokenParser(ll_interpreter, prompt_tokens)


    def get_logits(self, prompt: bytes, token_ids: list[int], media: Optional[dict] = None):
        """Computes the logits for the given token state.

        This overrides a method from the LocalEngine class that is used to get
        inference results from the model.
        """

        # make sure we don't run off the end of the model
        if len(token_ids) >= getattr(self.model_obj.config, "max_position_embeddings", 1e10):
            raise Exception(
                f"Attempted to run a transformers model past its maximum context window size of {self.model_obj.config.max_position_embeddings}!"
            )

        # get the number of cache positions we are using
        cache_token_ids = self._cached_token_ids
        num_cached = 0
        for id in cache_token_ids:
            if (
                num_cached >= len(cache_token_ids)
                or num_cached >= len(token_ids)
                or token_ids[num_cached] != id
            ):
                break
            num_cached += 1

        # reset the cache length according to that number of positions
        past_key_values = self._past_key_values
        past_length = past_key_values[0][0].size(-2) if past_key_values is not None else 0
        if past_length > num_cached:
            # note we recompute the last token because we don't bother to handle the special case of just computing logits
            past_length = max(0, num_cached - 1)
            self._past_key_values = tuple(
                tuple(p[..., :past_length, :] for p in v) for v in past_key_values
            )
        cache_token_ids[past_length:] = []
        new_token_ids = token_ids[past_length:]

        # If we had an input_processor, then model_inputs will be set
        model_inputs = {} if self.model_inputs is None else self.model_inputs
        model_inputs["input_ids"] = torch.tensor(new_token_ids).unsqueeze(0).to(self.device)
        model_inputs["attention_mask"]=torch.ones(1, past_length + len(new_token_ids)).to(self.device)

        # call the model
        if len(new_token_ids) > 0:
            with torch.no_grad():
                # Not all models support batched tokens for some reason
                try:
                    model_out = self.model_obj(
                        **model_inputs,
                        past_key_values=self._past_key_values,
                        use_cache=True,
                        position_ids=torch.arange(past_length, past_length + len(new_token_ids))
                        .unsqueeze(0)
                        .to(self.device),
                        return_dict=True,
                        output_attentions=False,
                        output_hidden_states=False,
                    )
                except AssertionError:
                    for i, new_token_id in enumerate(new_token_ids):
                        input_ids = torch.tensor([new_token_id]).unsqueeze(0).to(self.device)

                        model_out = self.model_obj(
                            input_ids=input_ids,
                            past_key_values=self._past_key_values,
                            use_cache=True,
                            position_ids=torch.arange(past_length, past_length + 1)
                            .unsqueeze(0)
                            .to(self.device),
                            attention_mask=torch.ones(1, past_length + 1).to(self.device),
                            return_dict=True,
                            output_attentions=False,
                            output_hidden_states=False,
                        )

                        self._past_key_values = model_out.past_key_values
                        past_length += 1

            # save the results
            self._past_key_values = model_out.past_key_values
            cache_token_ids.extend(new_token_ids)
            # Need to add special truncating logic here for weird models that have a different output size than tokenizer vocab
            self._cached_logits = (
                model_out.logits[0, -1, : len(self.tokenizer.tokens)].cpu().numpy()
            )
            self.metrics.engine_input_tokens += len(new_token_ids)
            self.metrics.engine_output_tokens += 1

        return self._cached_logits


class Transformers(Model):
    def __init__(
        self,
        model=None,
        tokenizer=None,
        echo=True,
        compute_log_probs=False,
        chat_template=None,
        **kwargs,
    ):
        """Build a new Transformers model object that represents a model in a given state."""
        if model == "microsoft/Phi-3-vision-128k-instruct":
            super().__init__(
                TransformersPhi3VisionEngine(
                    model,
                    compute_log_probs,
                    **kwargs
                ),
                echo=echo
            )
        else:
            super().__init__(
                TransformersEngine(
                    model,
                    tokenizer,
                    compute_log_probs,
                    chat_template=chat_template,
                    **kwargs,
                ),
                echo=echo,
            )


# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

import math
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor
from torch.nn import ModuleList
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
    TextStreamer,
)
from transformers.generation.utils import GenerateOutput

from .config import Settings
from .utils import batchify, empty_cache, print

# Import vLLM backend, but make it optional
try:
    from .vllm_backend import VLLMInferenceBackend

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    VLLMInferenceBackend = None


@dataclass
class AbliterationParameters:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float


class Model:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.vllm_backend = None
        self.current_dtype = None

        print()
        print(f"Loading model [bold]{settings.model}[/]...")

        # Validate inference backend setting
        if settings.inference_backend not in ["transformers", "vllm"]:
            raise ValueError(
                f"Invalid inference_backend: {settings.inference_backend}. "
                "Must be 'transformers' or 'vllm'."
            )

        if settings.inference_backend == "vllm" and not VLLM_AVAILABLE:
            print(
                "[yellow]Warning: vLLM backend requested but vLLM is not installed. "
                "Falling back to transformers backend.[/]"
            )
            settings.inference_backend = "transformers"

        if settings.inference_backend == "vllm":
            print(
                "* Using [bold]vLLM[/] backend for inference (faster, especially for AWQ models)"
            )
        else:
            print("* Using [bold]transformers[/] backend for inference")

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            settings.model
        )

        # Fallback for tokenizers that don't declare a special pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

        self.model = None

        for dtype in settings.dtypes:
            print(f"* Trying dtype [bold]{dtype}[/]... ", end="")

            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    settings.model,
                    dtype=dtype,
                    device_map=settings.device_map,
                )
                self.current_dtype = dtype

                # A test run can reveal dtype-related problems such as the infamous
                # "RuntimeError: probability tensor contains either `inf`, `nan` or element < 0"
                # (https://github.com/meta-llama/llama/issues/380).
                self.generate(["Test"], max_new_tokens=1)
            except Exception as error:
                self.model = None
                self.current_dtype = None
                empty_cache()
                print(f"[red]Failed[/] ({error})")
                continue

            print("[green]Ok[/]")
            break

        if self.model is None:
            raise Exception("Failed to load model with all configured dtypes.")

        # Check if model appears to be quantized and warn user
        if (
            settings.evaluate_model is None
        ):  # Only check during abliteration, not evaluation
            try:
                first_layer = self.get_layers()[0]
                # Check for common quantization indicators
                if (
                    hasattr(first_layer.self_attn.o_proj, "qweight")
                    or hasattr(first_layer.self_attn.o_proj, "qzeros")
                    or "awq" in settings.model.lower()
                    or "gptq" in settings.model.lower()
                ):
                    print()
                    print(
                        "[yellow]Warning: This model appears to be quantized (AWQ/GPTQ).[/]"
                    )
                    print(
                        "[yellow]Abliteration of quantized models may not work correctly.[/]"
                    )
                    print(
                        "[yellow]Consider using the base (non-quantized) version of this model instead.[/]"
                    )
                    print()
            except Exception:
                # If check fails, continue silently
                pass

        print(f"* Transformer model with [bold]{len(self.get_layers())}[/] layers")
        print("* Abliterable components:")
        for component, matrices in self.get_layer_matrices(0).items():
            print(
                f"  * [bold]{component}[/]: [bold]{len(matrices)}[/] matrices per layer"
            )

    def reload_model(self):
        # Keep the current dtype for consistency (already stored as string)
        reload_dtype = self.current_dtype if self.current_dtype else "auto"

        # Clean up vLLM backend if it exists
        if self.vllm_backend is not None:
            self.vllm_backend.cleanup()
            self.vllm_backend = None
            empty_cache()

        # Purge existing model object from memory to make space.
        self.model = None
        empty_cache()

        self.model = AutoModelForCausalLM.from_pretrained(
            self.settings.model,
            dtype=reload_dtype,
            device_map=self.settings.device_map,
        )
        # current_dtype remains the same as before

    def initialize_vllm_backend(self, model_path: str | None = None):
        """
        Initialize vLLM backend for inference.

        This should be called after abliteration is complete to enable
        fast inference with the modified model.

        Args:
            model_path: Path to the model (if None, uses settings.model)
        """
        if self.settings.inference_backend != "vllm" or not VLLM_AVAILABLE:
            return

        if model_path is None:
            model_path = self.settings.model

        # Clean up existing vLLM backend
        if self.vllm_backend is not None:
            self.vllm_backend.cleanup()
            self.vllm_backend = None
            empty_cache()

        try:
            print("* Initializing vLLM backend for inference...")
            # Use base model for tokenizer since tokenizer doesn't change during abliteration
            # The abliterated model and base model share the same tokenizer
            self.vllm_backend = VLLMInferenceBackend(
                model_path=model_path,
                tokenizer_path=self.settings.model,
                dtype=self.current_dtype if self.current_dtype else "auto",
                device_map=self.settings.device_map,
                quantization=self.settings.quantization,
                gpu_memory_utilization=self.settings.vllm_gpu_memory_utilization,
                max_model_len=self.settings.vllm_max_model_len,
            )
            print("  * [green]vLLM backend initialized successfully[/]")
            if self.settings.quantization:
                print(f"  * Using quantization: [bold]{self.settings.quantization}[/]")
            print(
                f"  * GPU memory utilization: [bold]{self.settings.vllm_gpu_memory_utilization:.0%}[/]"
            )
        except Exception as error:
            print(f"  * [yellow]Warning: Failed to initialize vLLM backend: {error}[/]")
            print("  * [yellow]Falling back to transformers backend[/]")
            self.vllm_backend = None

    def get_layers(self) -> ModuleList:
        # Most multimodal models.
        with suppress(Exception):
            return self.model.model.language_model.layers

        # Text-only models.
        return self.model.model.layers

    def get_layer_matrices(self, layer_index: int) -> dict[str, list[Tensor]]:
        layer = self.get_layers()[layer_index]

        matrices = {}

        def try_add(component: str, matrix: Any):
            # Handle Triton tensors (e.g., from MXFP4 quantization) by extracting
            # the underlying PyTorch tensor via the .data attribute.
            if hasattr(matrix, "data") and torch.is_tensor(matrix.data):
                matrix = matrix.data

            assert torch.is_tensor(matrix)

            if component not in matrices:
                matrices[component] = []

            matrices[component].append(matrix)

        # Exceptions aren't suppressed here, because there is currently
        # no alternative location for the attention out-projection.
        try_add("attn.o_proj", layer.self_attn.o_proj.weight)

        # Most dense models.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.mlp.down_proj.weight)

        # Some MoE models (e.g. Qwen3).
        with suppress(Exception):
            for expert in layer.mlp.experts:
                try_add("mlp.down_proj", expert.down_proj.weight)

        # Phi-3.5-MoE (and possibly others).
        with suppress(Exception):
            for expert in layer.block_sparse_moe.experts:
                try_add("mlp.down_proj", expert.w2.weight)

        # gpt-oss MoE.
        with suppress(Exception):
            # The implementation of gpt-oss in Transformers differs from many other MoE models
            # in that it stores the down-projections for all experts in a single 3D tensor,
            # but thanks to PyTorch's broadcasting magic, it all just works anyway.
            try_add("mlp.down_proj", layer.mlp.experts.down_proj)

        # Granite MoE Hybrid - attention layers with shared_mlp.
        with suppress(Exception):
            try_add("mlp.down_proj", layer.shared_mlp.output_linear.weight)

        # Granite MoE Hybrid - MoE layers with experts.
        with suppress(Exception):
            for expert in layer.moe.experts:
                try_add("mlp.down_proj", expert.output_linear.weight)

        # We need at least one MLP down-projection.
        assert matrices["mlp.down_proj"]

        return matrices

    def get_abliterable_components(self) -> list[str]:
        return list(self.get_layer_matrices(0).keys())

    def abliterate(
        self,
        refusal_directions: Tensor,
        direction_index: float | None,
        parameters: dict[str, AbliterationParameters],
    ):
        """
        Modify model weights to remove refusal behavior.

        Note: This method performs in-place weight modifications using matrix.sub_().
        It is designed for standard (non-quantized) models. Quantized models (AWQ/GPTQ)
        may not support in-place weight modifications and should be abliterated in their
        non-quantized form first, then quantized afterwards if needed.
        """
        if direction_index is None:
            refusal_direction = None
        else:
            # The index must be shifted by 1 because the first element
            # of refusal_directions is the direction for the embeddings.
            weight, index = math.modf(direction_index + 1)
            refusal_direction = F.normalize(
                refusal_directions[int(index)].lerp(
                    refusal_directions[int(index) + 1],
                    weight,
                ),
                p=2,
                dim=0,
            )

        # Note that some implementations of abliteration also orthogonalize
        # the embedding matrix, but it's unclear if that has any benefits.
        for layer_index in range(len(self.get_layers())):
            for component, matrices in self.get_layer_matrices(layer_index).items():
                params = parameters[component]

                distance = abs(layer_index - params.max_weight_position)

                # Don't orthogonalize layers that are more than
                # min_weight_distance away from max_weight_position.
                if distance > params.min_weight_distance:
                    continue

                # Interpolate linearly between max_weight and min_weight
                # over min_weight_distance.
                weight = params.max_weight + (distance / params.min_weight_distance) * (
                    params.min_weight - params.max_weight
                )

                if refusal_direction is None:
                    # The index must be shifted by 1 because the first element
                    # of refusal_directions is the direction for the embeddings.
                    layer_refusal_direction = refusal_directions[layer_index + 1]
                else:
                    layer_refusal_direction = refusal_direction

                # Projects any right-multiplied vector(s) onto the subspace
                # spanned by the refusal direction.
                projector = torch.outer(
                    layer_refusal_direction,
                    layer_refusal_direction,
                ).to(self.model.dtype)

                for matrix in matrices:
                    # Ensure projector is on the same device as the matrix for multi-GPU support.
                    device_projector = projector.to(matrix.device)
                    # In-place subtraction is safe as we're not using Autograd.
                    matrix.sub_(weight * (device_projector @ matrix))

    def get_chat(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.settings.system_prompt},
            {"role": "user", "content": prompt},
        ]

    def generate(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> tuple[BatchEncoding, GenerateOutput | LongTensor]:
        chats = [self.get_chat(prompt) for prompt in prompts]

        chat_prompts: list[str] = self.tokenizer.apply_chat_template(
            chats,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
        ).to(self.model.device)

        return inputs, self.model.generate(
            **inputs,
            **kwargs,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,  # Use greedy decoding to ensure deterministic outputs.
        )

    def get_responses(self, prompts: list[str]) -> list[str]:
        # Note: vLLM backend is only used when evaluating a pre-saved model,
        # not during the abliteration trials (where the model is modified in memory).
        # Use vLLM backend if available for faster inference
        if self.vllm_backend is not None:
            # Format prompts with chat template
            chats = [self.get_chat(prompt) for prompt in prompts]
            chat_prompts: list[str] = self.tokenizer.apply_chat_template(
                chats,
                add_generation_prompt=True,
                tokenize=False,
            )
            return self.vllm_backend.get_responses(
                chat_prompts,
                max_new_tokens=self.settings.max_response_length,
            )

        # Use transformers backend (default and for abliteration workflow)
        inputs, outputs = self.generate(
            prompts,
            max_new_tokens=self.settings.max_response_length,
        )

        # Return only the newly generated part.
        return self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

    def get_responses_batched(self, prompts: list[str]) -> list[str]:
        responses = []

        for batch in batchify(prompts, self.settings.batch_size):
            for response in self.get_responses(batch):
                responses.append(response)

        return responses

    def get_residuals(self, prompts: list[str]) -> Tensor:
        # We only generate one token, and we return the residual vectors
        # at that token position, for each prompt and layer.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        # Hidden states for the first (only) generated token.
        hidden_states = outputs.hidden_states[0]

        # The returned tensor has shape (prompt, layer, component).
        residuals = torch.stack(
            # layer_hidden_states has shape (prompt, position, component),
            # so this extracts the hidden states at the end of each prompt,
            # and stacks them up over the layers.
            [layer_hidden_states[:, -1, :] for layer_hidden_states in hidden_states],
            dim=1,
        )

        # Upcast the data type to avoid precision (bfloat16) or range (float16)
        # problems during calculations involving residual vectors.
        return residuals.to(torch.float32)

    def get_residuals_batched(self, prompts: list[str]) -> Tensor:
        residuals = []

        for batch in batchify(prompts, self.settings.batch_size):
            residuals.append(self.get_residuals(batch))

        return torch.cat(residuals, dim=0)

    # We work with logprobs rather than probabilities for numerical stability
    # when computing the KL divergence.
    def get_logprobs(self, prompts: list[str]) -> Tensor:
        # We only generate one token, and we return the (log) probability distributions
        # over the vocabulary at that token position, for each prompt.
        _, outputs = self.generate(
            prompts,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Logits for the first (only) generated token.
        logits = outputs.scores[0]

        # The returned tensor has shape (prompt, token).
        return F.log_softmax(logits, dim=-1)

    def get_logprobs_batched(self, prompts: list[str]) -> Tensor:
        logprobs = []

        for batch in batchify(prompts, self.settings.batch_size):
            logprobs.append(self.get_logprobs(batch))

        return torch.cat(logprobs, dim=0)

    def stream_chat_response(self, chat: list[dict[str, str]]) -> str:
        chat_prompt: str = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self.model.device)

        streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        outputs = self.model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=4096,
        )

        return self.tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

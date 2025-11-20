# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Philipp Emanuel Weidmann <pew@worldwidemann.com>

"""
vLLM backend for fast inference.

This module provides vLLM integration for faster inference, particularly beneficial
for AWQ quantized models. The abliteration process still uses transformers for
weight modification, but inference (generation and evaluation) can use vLLM.
"""

from typing import Any

from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput


class VLLMInferenceBackend:
    """
    Wrapper for vLLM inference engine.
    
    This backend is used for text generation during evaluation, while the
    transformers backend is still used for weight modification (abliteration).
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        dtype: str = "auto",
        device_map: str | dict = "auto",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        quantization: str | None = None,
    ):
        """
        Initialize vLLM backend.
        
        Args:
            model_path: Path or HuggingFace model ID
            tokenizer_path: Path or HuggingFace tokenizer ID
            dtype: Data type for model weights
            device_map: Device mapping (vLLM handles this automatically)
            gpu_memory_utilization: Fraction of GPU memory to use (default 0.9)
            max_model_len: Maximum sequence length (None = auto)
            quantization: Quantization format ('awq', 'gptq', etc.) - auto-detected if None
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        
        # Pass dtype to vLLM - it will validate and handle it appropriately
        vllm_dtype = dtype
        
        # Initialize vLLM engine with optional quantization
        llm_kwargs = {
            "model": model_path,
            "tokenizer": tokenizer_path,
            "dtype": vllm_dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
            "enforce_eager": False,  # Use CUDA graph for better performance
        }
        
        # Add quantization if specified
        if quantization is not None:
            llm_kwargs["quantization"] = quantization
        
        # Add max_model_len if specified
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        
        self.llm = LLM(**llm_kwargs)
        
        self.tokenizer = self.llm.get_tokenizer()

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 100,
        temperature: float = 0.0,  # Greedy decoding by default
        top_p: float = 1.0,
        top_k: int = -1,
        **kwargs: Any,
    ) -> list[RequestOutput]:
        """
        Generate text for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            **kwargs: Additional arguments (for compatibility)
            
        Returns:
            List of RequestOutput objects from vLLM
        """
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            skip_special_tokens=True,
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        return outputs

    def get_responses(
        self,
        prompts: list[str],
        max_new_tokens: int = 100,
    ) -> list[str]:
        """
        Generate responses for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            List of generated text responses (only the new tokens)
        """
        outputs = self.generate(prompts, max_new_tokens=max_new_tokens)
        
        # Extract only the generated text (without the prompt)
        responses = []
        for output in outputs:
            # vLLM returns only the generated text by default when skip_special_tokens=True
            generated_text = output.outputs[0].text
            responses.append(generated_text)
        
        return responses

    def cleanup(self):
        """Clean up vLLM resources explicitly."""
        if hasattr(self, 'llm'):
            # Free vLLM resources
            try:
                del self.llm
            except AttributeError:
                # Already cleaned up or never initialized
                pass
    
    def __del__(self):
        """Clean up vLLM resources."""
        self.cleanup()

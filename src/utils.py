import logging
from typing import Dict, List, Union

from transformers import AutoTokenizer
from vllm import LLM, RequestOutput, SamplingParams

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def run_inference(
    prompts: List[Dict[str, str]],
    llm: Union[str, LLM],
    tokenizer: AutoTokenizer = None,
    enable_thinking=False,
    tp_size=1,
    temperature=1.0,
    max_tokens=4096,
    n=1,
    gpu_memory_utilization=0.95,
    max_model_len=None,
    enable_expert_parallel=False,
    **kwargs,
) -> List[RequestOutput]:
    """Run inference on a list of prompts using a language model.

    Args:
        prompts (List[Dict[str, str]]): A list of prompts to generate responses for. Each prompt is expected to be in conversational format - a list of dictionaries with "role" and "content" keys.
        llm (Union[str, LLM]): The language model to use for inference. It is recommended to pass an LLM object as input to reuse it across multiple calls. If a string is passed, a new LLM object will be created using the model name.
        tokenizer (AutoTokenizer, optional): The tokenizer to use for encoding prompts. Defaults to None. If llm is a string, the tokenizer will be automatically created. If not, a tokenizer must be provided.
        enable_thinking (bool, optional): Whether to enable thinking mode. Defaults to False. Relevant only for the Qwen3 series of models.
        tp_size (int, optional): The tensor parallelism size. Defaults to 1.
        temperature (float, optional): The sampling temperature. Defaults to 1.0.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 4096.
        n (int, optional): The number of responses to generate. Defaults to 1.
        gpu_memory_utilization (float, optional): The GPU memory utilization ratio. Defaults to 0.95.
        enable_expert_parallel (bool, optional): Whether to enable expert parallelism. Defaults to False.
    Returns:
        List[RequestOutput]: A list of generated responses per prompt. Each request output can contain multiple generations if n > 1.
    """
    if tokenizer is None:
        if not isinstance(llm, str):
            raise ValueError("Tokenizer must be provided if llm is an instance of LLM")
        tokenizer = AutoTokenizer.from_pretrained(llm, trust_remote_code=True)
    if isinstance(llm, str):
        llm = LLM(
            model=llm,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_expert_parallel=enable_expert_parallel,
            max_model_len=max_model_len,
        )
    prompts = tokenizer.apply_chat_template(
        prompts,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=n,
        **kwargs,
    )
    log.info(f"Running inference on {len(prompts)} prompts")
    responses = llm.generate(prompts, sampling_params)
    return responses


def get_generated_text(responses: List[RequestOutput]) -> List[List[str]]:
    """Extract generated text from a list of RequestOutput objects.

    Args:
        responses (List[RequestOutput]): A list of RequestOutput objects containing generated responses.

    Returns:
        List[List[str]]: A list of lists containing the generated text for each prompt. The shape of the list is [num_prompts, num_responses_per_prompt].
    """
    return [[nth_response.text for nth_response in responses.outputs] for responses in responses]

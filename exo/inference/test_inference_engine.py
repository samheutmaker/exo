import asyncio
from dataclasses import dataclass
from typing import  List

from transformers import AutoTokenizer

from exo.inference.inference_engine import InferenceEngine
from exo.inference.mlx.sharded_inference_engine import MLXDynamicShardInferenceEngine
from exo.inference.mlx.sharded_utils import get_model_path
from exo.inference.shard import Shard

class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

def build_prompt(tokenizer, messages: List[Message]):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@dataclass
class Node:
    shard: Shard
    inference_engine: InferenceEngine


nodes = [
    Node(
        shard=Shard(
            model_id="mlx-community/Meta-Llama-3-8B-Instruct-4bit",
            start_layer=0,
            end_layer=7,
            n_layers=32,
        ),
        inference_engine=MLXDynamicShardInferenceEngine(),
    ),
    Node(
        shard=Shard(
            model_id="mlx-community/Meta-Llama-3-8B-Instruct-4bit",
            start_layer=8,
            end_layer=15,
            n_layers=32,
        ),
        inference_engine=MLXDynamicShardInferenceEngine(),
    ),
    Node(
        shard=Shard(
            model_id="mlx-community/Meta-Llama-3-8B-Instruct-4bit",
            start_layer=16,
            end_layer=23,
            n_layers=32,
        ),
        inference_engine=MLXDynamicShardInferenceEngine(),
    ),
    Node(
        shard=Shard(
            model_id="mlx-community/Meta-Llama-3-8B-Instruct-4bit",
            start_layer=24,
            end_layer=31,
            n_layers=32,
        ),
        inference_engine=MLXDynamicShardInferenceEngine(),
    ),
]


async def create_generation(prompt: str, model_id: str, nodes: List[Node]):
    model_path = await get_model_path(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompt = build_prompt(
        tokenizer,
        [Message(role="user", content=prompt)],
    )

    output_data, inference_state_full, _ = await nodes[0].inference_engine.infer_prompt(
        shard=nodes[0].shard,
        prompt=prompt,
    )

    is_finished = output_data.size == 1 and output_data.item() == tokenizer.eos_token_id

    result = ""
    node_index = 1

    while not is_finished:
        while node_index < len(nodes):
            node = nodes[node_index]
            node_index += 1

            (
                output_data,
                inference_state_full,
                _,
            ) = await node.inference_engine.infer_tensor(
                shard=node.shard,
                input_data=output_data,
                inference_state=inference_state_full,
            )

            is_finished = (
                output_data.size == 1 and output_data.item() == tokenizer.eos_token_id
            )

        if is_finished:
            break
        token = tokenizer.decode(output_data)
        result += token
        node_index = 0

    return result


# An inference engine should work the same for any number of Shards, as long as the Shards are continuous.
async def test_inference_engine():
    generation = await create_generation(
        "Tell me a story about a boy named billy?",
        "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        nodes,
    )

    print("DONE")
    print(generation)


asyncio.run(test_inference_engine())

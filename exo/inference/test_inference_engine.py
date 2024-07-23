import asyncio
import json

from dataclasses import dataclass, asdict
import pickle
from typing import List, Optional
import numpy as np

from transformers import AutoTokenizer
from nats import connect
from nats.aio.client import Client as NATS

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
class ForwardPassResult:
    output_data: np.ndarray
    inference_state_full: str
    is_finished: bool


@dataclass
class Node:
    model_id: str
    start_layer: int
    end_layer: int
    n_layers: int
    inference_engine: InferenceEngine
    nc: NATS
    shard: Shard = None
    tokenizer: AutoTokenizer = None

    is_root: bool = False
    is_leaf: bool = False
    topic: str = None
    next_topic: str = None

    def __post_init__(self):
        if self.shard is None:
            self.shard = Shard(
                self.model_id, self.start_layer, self.end_layer, self.n_layers
            )

        self.is_leaf = self.end_layer == self.n_layers - 1
        self.topic = f"shard_{self.start_layer}"
        self.next_topic = f"shard_{0 if self.is_leaf else self.end_layer + 1}"

        # Subscribe to the "topic" event on NATS
        asyncio.create_task(self.subscribe_to_topic())

    async def subscribe_to_topic(self):
        await self.nc.subscribe(self.topic, cb=self.handle_topic_message)
        print(f"Subscribed to {self.topic}")

    async def handle_topic_message(self, msg):
        # print(f"Received message on {msg.subject}")
        previous_forward_pass_result = pickle.loads(msg.data)
        next_forward_pass_result = await self.process_tensor(
            previous_forward_pass_result
        )

        if self.is_leaf:
            token = self.tokenizer.decode(next_forward_pass_result.output_data)
            print("token", token)

        if previous_forward_pass_result.is_finished:
            print("DONE")
            pass
        else:
            payload_bytes = pickle.dumps(next_forward_pass_result)
            # print(f"Sending message to {self.next_topic}")
            await self.nc.publish(self.next_topic, payload_bytes)

    async def process_prompt(self, prompt: str):
        (
            output_data,
            inference_state_full,
            is_finished,
        ) = await self.inference_engine.infer_prompt(
            shard=self.shard,
            prompt=prompt,
        )

        payload = ForwardPassResult(output_data, inference_state_full, is_finished)
        payload_bytes = pickle.dumps(payload)

        # print(f"Sending message to {self.next_topic}")
        response = await self.nc.request(self.next_topic, payload_bytes, timeout=20)
        return pickle.loads(response.data)

    async def process_tensor(self, previous_forward_pass_result: ForwardPassResult):
        (
            output_data,
            inference_state_full,
            is_finished,
        ) = await self.inference_engine.infer_tensor(
            shard=self.shard,
            input_data=previous_forward_pass_result.output_data,
            inference_state=previous_forward_pass_result.inference_state_full,
        )

        # print("is_finished", is_finished)

        return ForwardPassResult(output_data, inference_state_full, is_finished)


async def create_nodes(model_id: str, nc: NATS):
    model_path = await get_model_path(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    nodes = [
        Node(
            model_id=model_id,
            start_layer=0,
            end_layer=7,
            n_layers=32,
            inference_engine=MLXDynamicShardInferenceEngine(),
            nc=nc,
            tokenizer=tokenizer,
        ),
        Node(
            model_id=model_id,
            start_layer=8,
            end_layer=15,
            n_layers=32,
            inference_engine=MLXDynamicShardInferenceEngine(),
            nc=nc,
            tokenizer=tokenizer,
        ),
        Node(
            model_id=model_id,
            start_layer=16,
            end_layer=23,
            n_layers=32,
            inference_engine=MLXDynamicShardInferenceEngine(),
            nc=nc,
            tokenizer=tokenizer,
        ),
        Node(
            model_id=model_id,
            start_layer=24,
            end_layer=31,
            n_layers=32,
            inference_engine=MLXDynamicShardInferenceEngine(),
            nc=nc,
            tokenizer=tokenizer,
        ),
    ]

    return nodes, tokenizer


async def create_generation(prompt: str, nodes: List[Node], tokenizer: AutoTokenizer):
    prompt = build_prompt(
        tokenizer,
        [Message(role="user", content=prompt)],
    )

    import asyncio

    await asyncio.sleep(2)
    (output_data, inference_state_full, is_finished) = await nodes[0].process_prompt(
        prompt=prompt,
    )

    result = ""
    node_index = 1

    while not is_finished:
        while node_index < len(nodes):
            node = nodes[node_index]
            node_index += 1

            (
                output_data,
                inference_state_full,
                is_finished,
            ) = await node.process_tensor(
                input_data=output_data,
                inference_state=inference_state_full,
            )

        if is_finished:
            break
        token = tokenizer.decode(output_data)
        result += token
        node_index = 0

    return result


# An inference engine should work the same for any number of Shards, as long as the Shards are continuous.
async def test_inference_engine():
    nc = await connect("nats://localhost:4222")

    nodes, tokzenizer = await create_nodes(
        "mlx-community/Meta-Llama-3-8B-Instruct-4bit", nc
    )

    generation = await create_generation(
        "Tell me a story about a boy named billy?",
        nodes,
        tokzenizer,
    )

    print("DONE")
    print(generation)


asyncio.run(test_inference_engine())

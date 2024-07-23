import asyncio
import pickle
from dataclasses import dataclass
import numpy as np

from nats.aio.client import Client as NATS
from transformers import AutoTokenizer

from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard

STOP_SIGNAL = b"STOP"  # Use a byte string as the stop signal

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
    is_last: bool = False
    topic: str = None
    next_topic: str = None

    def __post_init__(self):
        if self.shard is None:
            self.shard = Shard(
                self.model_id, self.start_layer, self.end_layer, self.n_layers
            )

        self.is_last = self.end_layer == self.n_layers - 1
        self.topic = f"shard_{self.start_layer}"
        self.next_topic = f"shard_{0 if self.is_last else self.end_layer + 1}"

        # Subscribe to the "topic" event on NATS
        asyncio.create_task(self.subscribe_to_topic())

    async def subscribe_to_topic(self):
        await self.nc.subscribe(self.topic, cb=self.handle_topic_message)
        print(f"Subscribed to {self.topic}")

    async def handle_topic_message(self, msg):
        response_inbox = msg.headers.get("reply")
        previous_forward_pass_result = pickle.loads(msg.data)
        next_forward_pass_result = await self.process_tensor(
            previous_forward_pass_result
        )

        if next_forward_pass_result.is_finished:
            await self.nc.publish(response_inbox, STOP_SIGNAL)
            return

        if self.is_last:
            token = self.tokenizer.decode(next_forward_pass_result.output_data)
            token_bytes = token.encode("utf-8")
            await self.nc.publish(response_inbox, token_bytes)

        payload_bytes = pickle.dumps(next_forward_pass_result)

        await self.nc.publish(
            self.next_topic, payload_bytes, headers={"reply": response_inbox}
        )

    async def process_prompt(self, prompt: str) -> str:
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

        response_inbox = self.nc.new_inbox()

        sub = await self.nc.subscribe(response_inbox)

        await self.nc.publish(
            self.next_topic, payload_bytes, headers={"reply": response_inbox}
        )

        response = ""

        while True:
            try:
                msg = await sub.next_msg(
                    timeout=5.0  # Wait for 5 seconds for a message
                )
                token = None if msg.data == STOP_SIGNAL else msg.data.decode("utf-8")
                print(token)
                if token is None:
                    await sub.unsubscribe()
                    break
                else:
                    response += token
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error processing message: {e}")
                break

        return response

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

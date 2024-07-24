import argparse
import asyncio
import signal

from typing import List
from nats import connect
from nats.aio.client import Client as NATS
from transformers import AutoTokenizer

from exo.inference.mlx.sharded_inference_engine import MLXDynamicShardInferenceEngine
from exo.inference.mlx.sharded_utils import get_model_path
from exo.network.node import Node
from exo.api.chatgpt_api import build_prompt, Message

parser = argparse.ArgumentParser(description="Initialize GRPC Discovery")

parser.add_argument(
    "--shards",
    type=int,
    nargs="+",
    default=[1, 2, 3, 4],
    help="Array of shard boundaries",
)
args = parser.parse_args()


async def create_nodes(model_id: str, nc: NATS):
    model_path = await get_model_path(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    shard_params = [
        (0, 7),
        (8, 15),
        (16, 23),
        (24, 31),
    ]

    nodes = [
        Node(
            model_id=model_id,
            start_layer=start_layer,
            end_layer=end_layer,
            n_layers=32,
            inference_engine=MLXDynamicShardInferenceEngine(),
            nc=nc,
            tokenizer=tokenizer,
        )
        for i, (start_layer, end_layer) in enumerate(shard_params)
        if i + 1 in args.shards
    ]

    print("nodes.length", len(nodes))

    return nodes, tokenizer


async def shutdown(signal, loop):
    """Gracefully shutdown the server and close the asyncio loop."""
    server_tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [task.cancel() for task in server_tasks]
    print(f"Cancelling {len(server_tasks)} outstanding tasks")
    await asyncio.gather(*server_tasks, return_exceptions=True)
    # await server.stop()
    loop.stop()


async def create_generation(prompt: str, nodes: List[Node], tokenizer: AutoTokenizer):
    prompt = build_prompt(
        tokenizer,
        [Message(role="user", content=prompt)],
    )

    import asyncio

    await asyncio.sleep(2)

    return await nodes[0].process_prompt(
        prompt=prompt,
    )


async def main():
    nc = await connect("nats://100.84.82.82:4222")

    loop = asyncio.get_running_loop()

    # Use a more direct approach to handle signals
    def handle_exit():
        asyncio.ensure_future(shutdown(signal.SIGTERM, loop))

    for s in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(s, handle_exit)

    nodes, tokzenizer = await create_nodes(
        "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit", nc
    )

    if 1 in args.shards:
        generation = await create_generation(
            "Tell me a story about a boy named billy?",
            nodes,
            tokzenizer,
        )

    print(generation)

    await asyncio.Event().wait()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("Received keyboard interrupt. Shutting down...")
    finally:
        loop.run_until_complete(shutdown(signal.SIGTERM, loop))
        loop.close()

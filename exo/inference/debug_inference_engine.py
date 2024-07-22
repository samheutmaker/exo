from exo.inference.mlx.sharded_inference_engine import MLXDynamicShardInferenceEngine
from exo.inference.inference_engine import InferenceEngine
from exo.inference.shard import Shard
import asyncio
from exo.inference.mlx.sharded_utils import load_shard, get_model_path
import numpy as np
from mlx_lm.tokenizer_utils import load_tokenizer

# An inference engine should work the same for any number of Shards, as long as the Shards are continuous.
async def test_inference_engine(inference_engine_1: InferenceEngine, inference_engine_2: InferenceEngine, model_id: str):
    model_path = await get_model_path(model_id)
    _tokenizer = load_tokenizer(model_path)

    prompt = "In a single word only, what is the last name of the president of the United States? "
    resp_full, inference_state_full, _ = await inference_engine_1.infer_prompt(shard=Shard(model_id=model_id, start_layer=0, end_layer=31, n_layers=32), prompt=prompt)
    next_resp_full, next_inference_state_full, _ = await inference_engine_1.infer_tensor(shard=Shard(model_id=model_id, start_layer=0, end_layer=31, n_layers=32), input_data=resp_full, inference_state=inference_state_full)

    await inference_engine_1.reset_shard(shard=Shard(model_id=model_id, start_layer=0, end_layer=30, n_layers=32))
    resp1, inference_state_1, _ = await inference_engine_1.infer_prompt(shard=Shard(model_id=model_id, start_layer=0, end_layer=30, n_layers=32), prompt=prompt)

    await inference_engine_2.reset_shard(shard=Shard(model_id=model_id, start_layer=31, end_layer=31, n_layers=32))
    resp2, inference_state_2, _ = await inference_engine_2.infer_tensor(shard=Shard(model_id=model_id, start_layer=31, end_layer=31, n_layers=32), input_data=resp1, inference_state=inference_state_1)

    # don't reset the second time
    resp3, inference_state_3, _ = await inference_engine_1.infer_tensor(shard=Shard(model_id=model_id, start_layer=0, end_layer=30, n_layers=32), input_data=resp2, inference_state=inference_state_2)
    resp4, inference_state_4, _ = await inference_engine_2.infer_tensor(shard=Shard(model_id=model_id, start_layer=31, end_layer=31, n_layers=32), input_data=resp3, inference_state=inference_state_3)

    print(f"{resp2=}")
    print(f"full: {_tokenizer.decode(resp_full)}")
    print(f"next full: {_tokenizer.decode(next_resp_full)}")
    print(f"resp2: {_tokenizer.decode(resp2)}")
    print(f"{resp4=}")
    print(f"resp4: {_tokenizer.decode(resp4)}")

    assert np.array_equal(resp_full, resp2)
    assert np.array_equal(next_resp_full, resp4)


asyncio.run(
    test_inference_engine(
        MLXDynamicShardInferenceEngine(),
        MLXDynamicShardInferenceEngine(),
        "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
    )
)


# async def load_test():
#     shard = Shard(
#         model_id="mlx-community/Meta-Llama-3-8B-Instruct-4bit",
#         start_layer=0,
#         end_layer=31,
#         n_layers=32,
#     )

#     model, tokenizer = await load_shard(shard.model_id, shard)
#     print(f"Loaded {model}")
#     print(f"Loaded {tokenizer=}")

# asyncio.run(load_test())

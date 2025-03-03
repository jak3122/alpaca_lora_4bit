# Convert a huggingface LLaMA checkpoint to an (unsharded) pytorch checkpoint
# comes from https://github.com/tloen/alpaca-lora/blob/main/export_state_dict_checkpoint.py

import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("base_model")
parser.add_argument("size_key")
args = parser.parse_args()

tokenizer = LlamaTokenizer.from_pretrained(args.base_model)

base_model = LlamaForCausalLM.from_pretrained(
    args.base_model,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

base_model.train(False)

base_model_sd = base_model.state_dict()

params_by_model = {
    "7b": {
        "dim": 4096,
        "multiple_of": 256,
        "n_heads": 32,
        "n_layers": 32,
        "norm_eps": 1e-06,
        "vocab_size": -1,
    },
    "13b": {
        "dim": 5120,
        "multiple_of": 256,
        "n_heads": 40,
        "n_layers": 40,
        "norm_eps": 1e-06,
        "vocab_size": -1,
    },
    "30b": {
        "dim": 6656,
        "multiple_of": 256,
        "n_heads": 52,
        "n_layers": 60,
        "norm_eps": 1e-06,
        "vocab_size": -1,
    },
    "65b": {
        "dim": 8192,
        "multiple_of": 256,
        "n_heads": 64,
        "n_layers": 80,
        "norm_eps": 1e-06,
        "vocab_size": -1,
    },
}
params = params_by_model[args.size_key.lower()]
n_layers = params["n_layers"]
n_heads = params["n_heads"]
dim = params["dim"]
dims_per_head = dim // n_heads
base = 10000.0
inv_freq = 1.0 / (
    base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head)
)


def permute(w):
    return (
        w.view(n_heads, dim // n_heads // 2, 2, dim)
        .transpose(1, 2)
        .reshape(dim, dim)
    )


def unpermute(w):
    return (
        w.view(n_heads, 2, dim // n_heads // 2, dim)
        .transpose(1, 2)
        .reshape(dim, dim)
    )


def translate_state_dict_key(k):  # noqa: C901
    k = k.replace("base_model.model.", "")
    if k == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    elif k == "model.norm.weight":
        return "norm.weight"
    elif k == "lm_head.weight":
        return "output.weight"
    elif k.startswith("model.layers."):
        layer = k.split(".")[2]
        if k.endswith(".self_attn.q_proj.weight"):
            return f"layers.{layer}.attention.wq.weight"
        elif k.endswith(".self_attn.k_proj.weight"):
            return f"layers.{layer}.attention.wk.weight"
        elif k.endswith(".self_attn.v_proj.weight"):
            return f"layers.{layer}.attention.wv.weight"
        elif k.endswith(".self_attn.o_proj.weight"):
            return f"layers.{layer}.attention.wo.weight"
        elif k.endswith(".mlp.gate_proj.weight"):
            return f"layers.{layer}.feed_forward.w1.weight"
        elif k.endswith(".mlp.down_proj.weight"):
            return f"layers.{layer}.feed_forward.w2.weight"
        elif k.endswith(".mlp.up_proj.weight"):
            return f"layers.{layer}.feed_forward.w3.weight"
        elif k.endswith(".input_layernorm.weight"):
            return f"layers.{layer}.attention_norm.weight"
        elif k.endswith(".post_attention_layernorm.weight"):
            return f"layers.{layer}.ffn_norm.weight"
        elif k.endswith("rotary_emb.inv_freq") or "lora" in k:
            return None
        else:
            print(layer, k)
            raise NotImplementedError
    else:
        print(k)
        raise NotImplementedError


new_state_dict = {}
for k, v in base_model_sd.items():
    new_k = translate_state_dict_key(k)
    if new_k is not None:
        if "wq" in new_k or "wk" in new_k:
            new_state_dict[new_k] = unpermute(v)
        else:
            new_state_dict[new_k] = v

out_path = Path(args.base_model, "consolidated")
out_path.mkdir(exist_ok=True)
torch.save(new_state_dict, out_path / "consolidated.00.pth")

with open(out_path / "params.json", "w") as f:
    json.dump(params, f)

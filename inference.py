from amp_wrapper import AMPWrapper
import os
import sys
import argparse
import time
import torch
from autograd_4bit import load_llama_model_4bit_low_ram, load_llama_model_4bit_low_ram_and_offload_to_cpu, Autograd4bitQuantLinear
from transformers import TextStreamer

parser = argparse.ArgumentParser(
    prog=__file__.split(os.path.sep)[-1],
    description="Lora Inference",
)
parser.add_argument("--config_dir", default="llama-30b-4bit", required=False,
                    help="Path to the config.json, tokenizer_config.json, etc. Default: %(default)s"
                    )
parser.add_argument("--model_path", default="./llama-30b-4bit.pt", required=False,
                    help="Path to the quantized model in huggingface format. Default: %(default)s"
                    )
parser.add_argument("--lora_dir", default="lora", required=False,
                    help="Directory of fine-tuned lora results. Default: %(default)s"
                    )
parser.add_argument("--prompt", default="I think the meaning of life is", required=False,
                    help="Prompt for model inference. Default: %(default)s"
                    )
parser.add_argument("--n_tokens", default=200, type=int,
                    help="Number of tokens to generate. Default: %(default)s")
parser.add_argument("--temp", default=0.7, type=float, help="Temperature. Default: %(default)s")
parser.add_argument("--top_p", default=0.95, type=float, help="Top_p. Default: %(default)s")
parser.add_argument("--cpu", action="store_true", required=False,
                    help="CPU offloading. Default: %(default)s")
args = vars(parser.parse_args())

config_path = args['config_dir']
model_path = args['model_path']
lora_path = args['lora_dir']
prompt = args['prompt']
if args['cpu']:
    print('Loading model with CPU offload...')
    model, tokenizer = load_llama_model_4bit_low_ram_and_offload_to_cpu(
        config_path, model_path, lora_path=lora_path, groupsize=-1)
else:
    model, tokenizer = load_llama_model_4bit_low_ram(
        config_path, model_path, lora_path=lora_path, groupsize=-1)

print('Fitting 4bit scales and zeros to half')
model.half()
for n, m in model.named_modules():
    if isinstance(m, Autograd4bitQuantLinear):
        if m.groupsize == -1:
            m.zeros = m.zeros.half()
        m.scales = m.scales.half()
        m.bias = m.bias.half()

print('Apply AMP Wrapper ...')
wrapper = AMPWrapper(model)
wrapper.apply_generate()

batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
batch = {k: v.cuda() for k, v in batch.items()}
streamer = TextStreamer(tokenizer)

start = time.time()
with torch.no_grad():
    generated = model.generate(inputs=batch["input_ids"],
                               do_sample=True, use_cache=True,
                               repetition_penalty=1.1,
                               max_new_tokens=args["n_tokens"],
                               temperature=args["temp"],
                               top_p=args["top_p"],
                               top_k=20000,
                               return_dict_in_generate=True,
                               output_attentions=False,
                               output_hidden_states=False,
                               output_scores=False,
                               streamer=streamer)

result_tokens = generated['sequences'].cpu().tolist()[0]
result_text = tokenizer.decode(result_tokens)
end = time.time()
print(result_text)
print('Inference time: ', end - start, ' seconds')
print('ms per tokens: ', (end - start) / len(result_tokens) * 1000, ' ms')

import time

from transformers import AutoTokenizer

import mindspore as ms
from mindspore import JitConfig

from mindway.transformers import Qwen3ForCausalLM

ms.set_context(mode=0, jit_syntax_level=ms.STRICT)

start_time = time.time()

model_name = "QwQ/Qwen3-0.6B-250424"
model = Qwen3ForCausalLM.from_pretrained(
    model_name,
    mindspore_dtype=ms.bfloat16,
    attn_implementation="paged_attention",
)

# infer boost
jitconfig = JitConfig(jit_level="O0", infer_boost="on")
model.set_jit_config(jitconfig)

config = model.config
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("*" * 100)
print(f"Using {config._attn_implementation}, use_cache {config.use_cache},"
      f" dtype {config.mindspore_dtype}, layer {config.num_hidden_layers}")
print("Test passed: Sucessfully loaded Qwen3ForCausalLM")
print("Time elapsed: %.4fs" % (time.time() - start_time))
print("*" * 100)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

input_ids = ms.Tensor(tokenizer([text], return_tensors="np").input_ids, ms.int32)
model_inputs = {}
model_inputs["input_ids"] = input_ids

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=50,
    do_sample=False,
    use_cache=False,
)

generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(input_ids, generated_ids)]

outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(outputs)





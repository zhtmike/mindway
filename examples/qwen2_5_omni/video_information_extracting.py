import mindspore as ms
import numpy as np
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from mindway.transformers.models.qwen2_5_omni.qwen_omni_utils import process_mm_info

# inference function
def inference(video_path, prompt, sys_prompt):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,

                },
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # image_inputs, video_inputs = process_vision_info([messages])
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="np", padding=True, use_audio_in_video=False)
    # convert input to Tensor
    for key, value in inputs.items():  # by default input numpy array or list
        if isinstance(value, np.ndarray):
            inputs[key] = ms.Tensor(value)
        elif isinstance(value, list):
            inputs[key] = ms.Tensor(value)
        if inputs[key].dtype == ms.int64:
            inputs[key] = inputs[key].to(ms.int32)
        else:
            inputs[key] = inputs[key].to(model.dtype)

    output = model.generate(**inputs, use_audio_in_video=False, return_audio=False)

    text = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text


# Load model and processor
model_path = "Qwen/Qwen2.5-Omni-7B"
model = Qwen2_5OmniModel.from_pretrained(
    model_path,
    # mindspore_dtype=ms.bfloat16,
    attn_implementation="flash_attention_2",
)
processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

# Question 1
print("*"*100)
print("***** Question 1 *****")
video_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/shopping.mp4"
prompt = "How many kind of drinks can you see in the video?"

## Use a local model to inference.
response = inference(video_path, prompt=prompt, sys_prompt="You are a helpful assistant.")
print("***** Response 1 *****")
print(response[0])


print("*"*100)
print("***** Question 2 *****")
video_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/shopping.mp4"
prompt = "How many bottles of drinks have I picked up?"

## Use a local HuggingFace model to inference.
response = inference(video_path, prompt=prompt, sys_prompt="You are a helpful assistant.")
print("***** Response 2 *****")
print(response[0])


print("*"*100)
print("***** Question 3 *****")
video_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/shopping.mp4"
prompt = "How many milliliters are there in the bottle I picked up second time?"

## Use a local HuggingFace model to inference.
response = inference(video_path, prompt=prompt, sys_prompt="You are a helpful assistant.")
print("***** Response 3 *****")
print(response[0])
from transformers.utils.hub import cached_file
import torch
import mindspore as ms

def load_n_save_speakers(path, ms_path):
    speaker_map = {}
    for key, value in torch.load(path).items():
        if isinstance(value, dict):
            value_ms_dict = {}
            for k, v in value.items():
                if torch.is_tensor(v):
                    value_ms_dict[k] = ms.tensor(v.cpu().numpy())
                else:
                    value_ms_dict[k] = v
            speaker_map[key] = value_ms_dict
        else:
            if torch.is_tensor(value):
                speaker_map[key] = ms.tensor(value.cpu().numpy())
            else:
                speaker_map[key] = value
    print("Speaker torch2ms loaded:", speaker_map)
    print("Speaker {} loaded".format(list(speaker_map.keys())))

    ms.save_checkpoint(ms_path, speaker_map)

    # validate
    ms_dict = ms.load_checkpoint(ms_path)
    speaker_map = {}
    for key, value in ms_dict.items():
        speaker_map[key] = value
    print("Speaker ms loaded:", speaker_map)
    print("Speaker {} loaded".format(list(speaker_map.keys())))

spk_path="Qwen/Qwen2.5-Omni-7B/spk_dict.pt"
ms_spk_path="Qwen/Qwen2.5-Omni-7B/spk_dict.ckpt"

load_n_save_speakers(spk_path, ms_spk_path)
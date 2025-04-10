import torch
import numpy as np

def load_n_save_speakers(path, np_path):
    speaker_map = {}
    for key, value in torch.load(path).items():
        if isinstance(value, dict):
            value_ms_dict = {}
            for k, v in value.items():
                if torch.is_tensor(v):
                    value_ms_dict[k] = v.cpu().numpy()
                else:
                    value_ms_dict[k] = v
            speaker_map[key] = value_ms_dict
        else:
            if torch.is_tensor(value):
                speaker_map[key] = value.cpu().numpy()
            else:
                speaker_map[key] = value
    # print("Speaker torch2ms loaded:", speaker_map)
    print("Speaker {} loaded".format(list(speaker_map.keys())))

    np.save(np_path, speaker_map)

    # validate
    np_dict = np.load(np_path, allow_pickle=True).item()
    speaker_map = {}
    for key, value in np_dict.items():
        speaker_map[key] = value
    # print("Speaker ms loaded:", speaker_map)
    print("Speaker {} loaded".format(list(speaker_map.keys())))

spk_path="Qwen/Qwen2.5-Omni-7B/spk_dict.pt"
np_spk_path="Qwen/Qwen2.5-Omni-7B/spk_dict.npy"

load_n_save_speakers(spk_path, np_spk_path)
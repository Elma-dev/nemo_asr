# datasets_info = [
#     {
#         "ds_name": "atlasia/MoulSot-Full",
#         "name": "100-gt-2.5",
#         "ds_split": "train",
#         "text_column": "text",
#         "audio_cl": "audio",
#         "fl_split": "train",
#         "out_manifest": "train.json",
#     },
#     {
#         "ds_name": "atlasia/MoulSot-Full",
#         "name": "100-gt-2.5",
#         "ds_split": "test",
#         "text_column": "text",
#         "audio_cl": "audio",
#         "fl_split": "test",
#         "out_manifest": "test.json",
#     },
#     {
#         "ds_name": "atlasia/MoulSot-Full",  # here will be another dataset i think like mgb5
#         "name": "",
#         "ds_split": "val",
#         "text_column": "text",
#         "audio_cl": "audio",
#         "fl_split": "validation",
#         "out_manifest": "validation.json",
#     },
# ]

# MANIFESTS = ["train.json", "test.json", "validation.json"]

# TOKENIZER_DIR = "tokenizer"
# VOCAB_SIZE = 1024
# TOKENIZER = "spe"
# SPE_TYPE = "bpe"


# TRAIN_MANIFEST_FILEPATH = "train.json"
# VALIDATION_MANIFEST_FILEPATH = "test.json"
# CTC_LOSS_WEIGHT = 0.3
# DEVICES = -1
# MAX_EPOCHS = 100
# OPTIM_NAME = "adamw"
# lr = 0.001
# betas = [0.9, 0.999]
# weight_decay = 0.0001
# warmup_steps = 2000


# datasets_info = [
#     {
#         "ds_name": "atlasia/MoulSot-Full",
#         "name": "100-gt-2.5",
#         "ds_split": "train",
#         "text_column": "text",
#         "audio_cl": "audio",
#         "fl_split": "train",
#         "out_manifest": "train.json",
#     },
#     {
#         "ds_name": "atlasia/MoulSot-Full",
#         "name": "100-gt-2.5",
#         "ds_split": "test",
#         "text_column": "text",
#         "audio_cl": "audio",
#         "fl_split": "test",
#         "out_manifest": "test.json",
#     },
#     {
#         "ds_name": "atlasia/MoulSot-Full",  # here will be another dataset i think like mgb5
#         "name": "",
#         "ds_split": "val",
#         "text_column": "text",
#         "audio_cl": "audio",
#         "fl_split": "validation",
#         "out_manifest": "validation.json",
#     },
# ]

# MANIFESTS = ["train.json", "test.json", "validation.json"]

# TOKENIZER_DIR = "tokenizer"
# VOCAB_SIZE = 1024
# TOKENIZER = "spe"
# SPE_TYPE = "bpe"


# TRAIN_MANIFEST_FILEPATH = "train.json"
# VALIDATION_MANIFEST_FILEPATH = "test.json"
# CTC_LOSS_WEIGHT = 0.3
# DEVICES = -1
# MAX_EPOCHS = 100
# OPTIM_NAME = "adamw"
# lr = 0.001
# betas = [0.9, 0.999]
# weight_decay = 0.0001
# warmup_steps = 2000


datasets_info = [
    {
        "ds_name": "abdeljalilELmajjodi/Atlaset-audio",
        "name": "NADI2025_subtask2_ASR",
        "ds_split": "train",
        "text_column": "transcription",
        "audio_cl": "audio",
        "fl_split": "train",
        "out_manifest": "train.json",
        "size": 100,
    },
    {
        "ds_name": "atlasia/MoulSot-Full",
        "name": "",
        "ds_split": "test",
        "text_column": "text",
        "audio_cl": "audio",
        "fl_split": "test",
        "out_manifest": "test.json",
        "size": 100,
    },
]

MANIFESTS = ["train.json", "test.json"]

TOKENIZER_DIR = "tokenizer"
VOCAB_SIZE = 1024
TOKENIZER = "spe"
SPE_TYPE = "bpe"

TOKENIZER_BPE_DIR = f"tokenizer/tokenizer_spe_bpe_v{VOCAB_SIZE}"
TRAIN_MANIFEST_FILEPATH = "train.json"
VALIDATION_MANIFEST_FILEPATH = "test.json"
CTC_LOSS_WEIGHT = 0.3
DEVICES = -1
MAX_EPOCHS = 100
OPTIM_NAME = "adamw"
lr = 0.001
betas = "[0.9,0.999]"
weight_decay = 0.0001
warmup_steps = 2000

WANDB_LOGGER = True
WANDB_EXP_NAME = "nemo_asr_exp"
WANDB_PRO_NAME = "moulsot_asr"

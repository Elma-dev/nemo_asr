from configs.configs import warmup_steps
from configs.configs import weight_decay
from configs.configs import betas
from configs.configs import OPTIM_NAME
from configs.configs import MAX_EPOCHS
from configs.configs import DEVICES
from configs.configs import CTC_LOSS_WEIGHT
from configs.configs import VALIDATION_MANIFEST_FILEPATH
from configs.configs import TRAIN_MANIFEST_FILEPATH
from configs.configs import (
    MANIFESTS,
    TOKENIZER_DIR,
    VOCAB_SIZE,
    TOKENIZER,
    SPE_TYPE,
    lr,
)
import subprocess
from utils.data_preparation import load_export_dataset
from configs import configs


def main():
    for di in configs.datasets_info:
        load_export_dataset(
            dataset_name=di["ds_name"],
            name=di["name"],
            dataset_split=di["ds_split"],
            text_column=di["text_column"],
            audio_column=di["audio_cl"],
            folder_split=di["fl_split"],
            out_manifest=di["out_manifest"],
        )

    tokenizer_cmd = f"""
    python utils.process_asr_text_tokenizer.py \
        --manifest {",".join([m for m in MANIFESTS])} \
        --data_root "{TOKENIZER_DIR}" \
        --vocab_size {VOCAB_SIZE} \
        --tokenizer {TOKENIZER} \
        --spe_type {SPE_TYPE} \
        --log
    """

    subprocess.run(tokenizer_cmd, shell=True, check=True)

    train_cmd = f"""python speech_to_text_hybrid_rnnt_ctc_bpe.py \
    model.train_ds.manifest_filepath={TRAIN_MANIFEST_FILEPATH} \
    model.validation_ds.manifest_filepath={VALIDATION_MANIFEST_FILEPATH} \
    model.tokenizer.dir={TOKENIZER_DIR} \
    model.tokenizer.type={SPE_TYPE} \
    model.aux_ctc.ctc_loss_weight={CTC_LOSS_WEIGHT} \
    trainer.devices={DEVICES} \
    trainer.max_epochs={MAX_EPOCHS} \
    model.optim.name={OPTIM_NAME} \
    model.optim.lr={lr} \
    model.optim.betas={betas} \
    model.optim.weight_decay={weight_decay} \
    model.optim.sched.warmup_steps={warmup_steps}"""

    subprocess.run(train_cmd, shell=True, check=True)

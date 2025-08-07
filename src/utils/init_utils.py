import logging
import os
import random
import secrets
import shutil
import string
import subprocess

import numpy as np
import torch
from omegaconf import OmegaConf

from src.logger.logger import setup_logging
from src.utils.io_utils import ROOT_PATH


def set_worker_seed(worker_id):
    """
    Set seed for each dataloader worker.

    For more info, see https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        worker_id (int): id of the worker.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_random_seed(seed):
    """
    Set random seed for model training or inference.

    Args:
        seed (int): defines which seed to use.
    """
    # fix random seeds for reproducibility
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=True works faster but reproducibility decreases
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# https://github.com/wandb/wandb/blob/main/wandb/sdk/lib/runid.py
def generate_id(length: int = 8) -> str:
    """
    Generate a random base-36 string of `length` digits.

    Args:
        length (int): length of a string.
    Returns:
        run_id (str): base-36 string with an experiment id.
    """
    # There are ~2.8T base-36 8-digit strings. If we generate 210k ids,
    # we'll have a ~1% chance of collision.
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def log_git_commit_and_patch(save_dir):
    """
    Log current git commit and patch to save dir.
    Improves reproducibility by allowing to run the same code version:
        git checkout commit_hash_from_commit_path
        git apply patch_path

    If you created new files and want to have them in patch,
    stage them via git add before running the script.

    Patch can be applied via the following command:
        git apply patch_path

    Args:
        save_dir (Path): directory to save patch and commit in
    """
    print("Logging git commit and patch...")
    commit_path = save_dir / "git_commit.txt"
    patch_path = save_dir / "git_diff.patch"
    with commit_path.open("w") as f:
        subprocess.call(["git", "rev-parse", "HEAD"], stdout=f)
    with patch_path.open("w") as f:
        subprocess.call(["git", "diff", "HEAD"], stdout=f)


def resume_config(save_dir):
    """
    Get run_id from resume config to continue logging
    to the same experiment.

    Args:
        save_dir (Path): path to the directory with the run config.
    Returns:
        run_id (str): base-36 string with experiment id.
    """
    saved_config = OmegaConf.load(save_dir / "config.yaml")
    run_id = saved_config.writer.run_id
    print(f"Resuming training from run {run_id}...")
    return run_id


def saving_init(save_dir, config):
    """
    Initialize saving by getting run_id.

    Args:
        save_dir (Path): path to the directory to log everything:
            logs, checkpoints, config, etc.
        config (DictConfig): hydra config for the current experiment.
    """
    run_id = None

    if save_dir.exists():
        if config.trainer.get("resume_from") is not None:
            run_id = resume_config(save_dir)
        elif config.trainer.override:
            print(f"Overriding save directory '{save_dir}'...")
            shutil.rmtree(str(save_dir))
        elif not config.trainer.override:
            raise ValueError(
                "Save directory exists. Change the name or set override=True"
            )

    save_dir.mkdir(exist_ok=True, parents=True)

    if run_id is None:
        run_id = generate_id(length=config.writer.id_length)

    OmegaConf.set_struct(config, False)
    config.writer.run_id = run_id
    OmegaConf.set_struct(config, True)

    OmegaConf.save(config, save_dir / "config.yaml")

    log_git_commit_and_patch(save_dir)


def setup_saving_and_logging(config):
    """
    Initialize the logger, writer, and saving directory.
    The saving directory is defined by the run_name and save_dir
    arguments of config.writer and config.trainer, respectfully.

    Args:
        config (DictConfig): hydra config for the current experiment.
    Returns:
        logger (Logger): logger that logs output.
    """
    save_dir = ROOT_PATH / config.trainer.save_dir / config.writer.run_name
    saving_init(save_dir, config)

    if config.trainer.get("resume_from") is not None:
        setup_logging(save_dir, append=True)
    else:
        setup_logging(save_dir, append=False)
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)

    return logger


def convOutputSizeParam(input_size, kernel_size, stride, padding=0):
    return [int(input_size[0] - kernel_size + 2 * padding / stride) + 1, 
            int(input_size[1] - kernel_size + 2 * padding / stride) + 1]

def maxPoolOutputSizeParam(input_size, filter_size, stride):
    return [int((input_size[0] - filter_size) / stride) + 1,
            int((input_size[1] - filter_size) / stride) + 1]

def computeLinearInputSize(input_audio):
    size = list(input_audio.size())[1:] #CHANNELS x WIDTH x HEIGHT torch.Size([1, 60, 750])
    size_1 = convOutputSizeParam(input_size=size, kernel_size=5, stride=1, padding=2)
    size_2 = maxPoolOutputSizeParam(input_size=size_1, filter_size=2, stride=2)
    size_3 = convOutputSizeParam(input_size=size_2, kernel_size=1, stride=1)
    #MFM preserves WIDTH x HEIGHT
    #BatchNorm preserves WIDTH x HEIGHT
    size_7 = convOutputSizeParam(input_size=size_3, kernel_size=3, stride=1, padding=1)
    #MFM preserves WIDTH x HEIGHT
    size_9 = maxPoolOutputSizeParam(input_size=size_7, filter_size=2, stride=2)
    #BatchNorm preserves WIDTH x HEIGHT
    size_11 = convOutputSizeParam(input_size=size_9, kernel_size=1, stride=1)
    #MFM preserves WIDTH x HEIGHT
    #BatchNorm preserves WIDTH x HEIGHT
    size_14 = convOutputSizeParam(input_size=size_11, kernel_size=3, stride=1, padding=1)
    #MFM preserves WIDTH x HEIGHT
    size_16 = maxPoolOutputSizeParam(input_size=size_14, filter_size=2, stride=2)
    size_17 = convOutputSizeParam(input_size=size_16, kernel_size=1, stride=1)
    #MFM preserves WIDTH x HEIGHT
    #BatchNorm preserves WIDTH x HEIGHT
    size_20 = convOutputSizeParam(input_size=size_17, kernel_size=3, stride=1, padding=1)
    #MFM preserves WIDTH x HEIGHT
    #BatchNorm preserves WIDTH x HEIGHT
    size_23 = convOutputSizeParam(input_size=size_20, kernel_size=1, stride=1)
    #MFM preserves WIDTH x HEIGHT
    #BatchNorm preserves WIDTH x HEIGHT
    size_26 = convOutputSizeParam(input_size=size_23, kernel_size=3, stride=1, padding=1)
    #MFM preserves WIDTH x HEIGHT
    size_28 = maxPoolOutputSizeParam(input_size=size_26, filter_size=2, stride=2)
    return size_28
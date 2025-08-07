import random
import torch

def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    K = 750 # In https://arxiv.org/pdf/2103.11326 it is claimed that it is enough to fix length of k=750 to cover input features of 98% trials.
    result_batch = {"data_object": [], "label": []}

    for elem in dataset_items:
        audio = elem["data_object"]
        _, _, length = audio.shape

        if length < K:
            repeat_factor = (K + length - 1) // length
            audio = audio.repeat(1, 1, repeat_factor)
            audio = audio[:, :, :K]
        else:
            max_start = length - K
            start = random.randint(0, max_start)
            audio = audio[:, :, start:start + K]
        
        result_batch["data_object"].append(audio)
        result_batch["label"].append(elem["label"])

    result_batch["data_object"] = torch.stack(result_batch["data_object"])
    result_batch["label"] = torch.tensor(result_batch["label"])

    return result_batch

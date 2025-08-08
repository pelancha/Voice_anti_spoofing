import torchaudio

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json

from pathlib import Path


# class ExampleDataset(BaseDataset):
#     """
#     Example of a nested dataset class to show basic structure.

#     Uses random vectors as objects and random integers between
#     0 and n_classes-1 as labels.
#     """

#     def __init__(
#         self, input_length, n_classes, dataset_length, name="train", *args, **kwargs
#     ):
#         """
#         Args:
#             input_length (int): length of the random vector.
#             n_classes (int): number of classes.
#             dataset_length (int): the total number of elements in
#                 this random dataset.
#             name (str): partition name
#         """
#         index_path = ROOT_PATH / "data" / "example" / name / "index.json"

#         # each nested dataset class must have an index field that
#         # contains list of dicts. Each dict contains information about
#         # the object, including label, path, etc.
#         if index_path.exists():
#             index = read_json(str(index_path))
#         else:
#             index = self._create_index(input_length, n_classes, dataset_length, name)

#         super().__init__(index, *args, **kwargs)

#     def _create_index(self, input_length, n_classes, dataset_length, name):
#         """
#         Create index for the dataset. The function processes dataset metadata
#         and utilizes it to get information dict for each element of
#         the dataset.

#         Args:
#             input_length (int): length of the random vector.
#             n_classes (int): number of classes.
#             dataset_length (int): the total number of elements in
#                 this random dataset.
#             name (str): partition name
#         Returns:
#             index (list[dict]): list, containing dict for each element of
#                 the dataset. The dict has required metadata information,
#                 such as label and object path.
#         """
#         index = []
#         data_path = ROOT_PATH / "data" / "example" / name
#         data_path.mkdir(exist_ok=True, parents=True)

#         # to get pretty object names
#         number_of_zeros = int(np.log10(dataset_length)) + 1

#         # In this example, we create a synthesized dataset. However, in real
#         # tasks, you should process dataset metadata and append it
#         # to index. See other branches.
#         print("Creating Example Dataset")
#         for i in tqdm(range(dataset_length)):
#             # create dataset
#             example_path = data_path / f"{i:0{number_of_zeros}d}.pt"
#             example_data = torch.randn(input_length)
#             example_label = torch.randint(n_classes, size=(1,)).item()
#             torch.save(example_data, example_path)

#             # parse dataset metadata and append it to index
#             index.append({"path": str(example_path), "label": example_label})

#         # write index to disk
#         write_json(index, str(data_path / "index.json"))

#         return index


class AntiSpoofDataset(BaseDataset):
    def __init__(
        self, name="train", *args, **kwargs
    ):
        """
        Args:
            input_length (int): length of the random vector.
            n_classes (int): number of classes.
            dataset_length (int): the total number of elements in
                this random dataset.
            name (str): partition name
        """
        index_dir_path = ROOT_PATH / "data" / "example" / name
        index_path = index_dir_path / "index.json"

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(index_dir_path, name)

        super().__init__(index, *args, **kwargs)


    def _create_index(self, index_dir_path, name):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.
        """
        index = []
        data_path = Path("/kaggle") / "input" / "asvpoof-2019-dataset" / "LA" / "LA" / "ASVspoof2019_LA_cm_protocols" / f"ASVspoof2019.LA.cm.{name}.tr{'n' if name == 'train' else 'l'}.txt"
        audio_path = Path("/kaggle") / "input" / "asvpoof-2019-dataset" / "LA" / "LA" / f"ASVspoof2019_LA_{name}" / "flac"

        print(f"Data path: {data_path}")
        print(f"Audio path: {audio_path}")

        index_dir_path.mkdir(parents=True, exist_ok=True)

        print(f"Reading ASVspoof2019 {name} dataset...")
        with open(data_path, 'r') as f:
            for line in f:
                line = line.split()
                # parse dataset metadata and append it to index
                index.append({"path": str(audio_path / f"{line[1]}.flac"), 
                              "key": f"{line[1]}",
                              "label": 1 if line[-1] == "bonafide" else 0})

        # write index to disk
        write_json(index, str(index_dir_path / "index.json"))

        return index
    
    #TODO: should it be changed here or in base class? will def from base class grab this overriding?
    def load_object(self, path):
            """
            Load object from disk.

            Args:
                path (str): path to the object.
            Returns:
                data_object (Tensor):
            """
            data_object, _ = torchaudio.load(path) #TODO: check if it works without torchaudio - its initial variant from base class
            return data_object
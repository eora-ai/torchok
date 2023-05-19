
from pathlib import Path
from functools import partial
from typing import List, Dict, Any, Iterable

import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from torchok.data.datasets.classification.classification import process_multilabel


def batching(iterable: Iterable, batch_size: int = 64):
    iter_length = len(iterable)
    for ndx in range(0, iter_length, batch_size):
        yield iterable[ndx: min(ndx + batch_size, iter_length)]


class ClassificationNLPFineTuneDataset(Dataset):
    TARGET_TYPES = ['multiclass', 'multilabel', 'embedding']

    def __init__(self,
                 data_folder: str,
                 annotation_path: str,
                 targets: List[Dict[str, Any]],
                 input_column: str = 'text',
                 lazy_init: bool = False,
                 test_mode: bool = False):
        """"
        targets: List of dicts where each dict contain information about heads
            - `name` (str): Name of the output target.
            - `column` (str): Column name containing image label.
            - `type` (str): Format of data processing. Available values: multiclass, multilabel, embedding.
            - `num_classes` (int): Number of classes.
            - `dtype` (str): Data type of the torch tensors related to the target.
            - `path_to_embeddings` (str): Used only when `target_type` is 'embedding',
                path to .npy with embeddings per each image.
        """
        super().__init__()
        self.data_folder = Path(data_folder)
        self.annotation_path = annotation_path
        self.input_column = input_column
        self.lazy_init = lazy_init
        self.test_mode = test_mode

        if annotation_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_folder / annotation_path)
        elif annotation_path.endswith('.pkl'):
            self.df = pd.read_pickle(self.data_folder / annotation_path)
        else:
            raise ValueError('Detection dataset error. Annotation path is not in `csv` or `pkl` format')

        self.heads = []
        if not self.test_mode:
            for target in targets:
                name = target['name']
                column = target['column']
                target_type = target['type']
                num_classes = target.get('num_classes', None)
                target_dtype = target.get('dtype', 'long')

                if target_type == 'multiclass':
                    self.heads.append((name, column, target_type, num_classes, target_dtype))
                elif target_type == 'multilabel':
                    self.heads.append((name, column, target_type, num_classes, target_dtype))
                    self.df[column] = self.df[column].fillna('')
                    if not self.lazy_init:
                        self.df[column] = self.df[column].apply(partial(process_multilabel, num_classes=num_classes))
                elif target_type == 'embedding':
                    self.heads.append((name, column, target_type, num_classes, target_dtype))
                    data = np.load(self.data_folder / target['path_to_embeddings'], allow_pickle=True)
                    self.df[column] = list(data)
                else:
                    raise ValueError(f'This target {target_type} type is not supported')

    def __len__(self) -> int:
        """Dataset length."""
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        record = self.df.iloc[idx]
        sample = {'index': idx}

        if not self.test_mode:
            for name, column, target_type, num_classes, target_dtype in self.heads:
                label = record[column]
                if target_type == 'multilabel' and self.lazy_init:
                    label = process_multilabel(label, num_classes)
                sample[f'target_{name}'] = torch.tensor(label).type(torch.__dict__[target_dtype])

        return sample


class ClassificationNLPFineTuneWithEmbeddingGenerationDataset(ClassificationNLPFineTuneDataset):
    def __init__(self,
                 data_folder: str,
                 annotation_path: str,
                 targets: List[Dict[str, Any]],
                 sentence_transformer_name: str,
                 generate_embeddings: bool = True,
                 generated_embeddings_column_name: str = "embedding",
                 save_prefix: str = None,
                 input_column: str = 'text',
                 lazy_init: bool = False,
                 test_mode: bool = False):
        super().__init__(data_folder=data_folder,
                         annotation_path=annotation_path,
                         targets=targets,
                         input_column=input_column,
                         lazy_init=lazy_init,
                         test_mode=test_mode)

        self.save_name = sentence_transformer_name.split('/')[-1] + "embedding"
        if save_prefix is not None:
            self.save_name = save_prefix + "_" + self.save_name

        if generate_embeddings:
            sentences = self.df[self.input_column].tolist()

            if Path(self.save_name).is_file():
                raise ValueError("Add `save_prefix` to prevent delete method for already exist file.")

            embeddings = self.save_embeddings(sentences=sentences,
                                              model_name=sentence_transformer_name,
                                              data_folder=data_folder,
                                              save_name=self.save_name)

            self.heads.append((generated_embeddings_column_name,
                               generated_embeddings_column_name,
                               generated_embeddings_column_name,
                               None,
                               "float32"))
            self.df[generated_embeddings_column_name] = list(embeddings)


    @staticmethod
    def save_embeddings(sentences: list[str], model_name: str, data_folder: str, save_name: str) -> np.ndarray:
        model = SentenceTransformer(model_name)

        embeddings = []
        for batch in batching(sentences):
            embeddings += model.encode(batch).tolist()
        embeddings = np.array(embeddings, dtype=np.float32)

        save_path = Path(data_folder) / save_name
        np.save(save_path.as_posix(), embeddings)

        del model
        torch.cuda.empty_cache()

        return embeddings

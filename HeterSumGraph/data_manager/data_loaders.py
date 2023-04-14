from concurrent.futures import ThreadPoolExecutor

from torch.utils.data import BatchSampler, IterDataPipe, MapDataPipe
from torch.utils.data._utils.fetch import _BaseDatasetFetcher
from torch.utils.data.dataloader import _BaseDataLoaderIter, _DatasetKind, _MultiProcessingDataLoaderIter
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES

from module.dataloader import SummarizationDataSet, graph_collate_fn
import torch


class _MapDatasetFetcher(_BaseDatasetFetcher):
    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                data = self.dataset.__getitems__(possibly_batched_index)
            else:
                data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self.collate_fn(data)



class DataLoader(torch.utils.data.DataLoader):
    pass



def make_dataloader(data_file, vocab, hps, filter_word, w2s_path, graphs_dir, max_instance=None):
    dataset = SummarizationDataSet(data_file, vocab, hps.doc_max_timesteps, hps.sent_max_len, filter_word, w2s_path,
                                   max_instance=max_instance, graphs_dir=graphs_dir)
    loader = DataLoader(dataset, batch_size=hps.batch_size, shuffle=False, num_workers=0,
                        collate_fn=graph_collate_fn)
    del dataset
    return loader

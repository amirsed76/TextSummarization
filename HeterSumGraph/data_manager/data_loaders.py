from torch.utils.data import BatchSampler

from module.dataloader import SummarizationDataSet, graph_collate_fn,CachedSummarizationDataSet
import torch


def make_dataloader(data_file, vocab, hps, filter_word, w2s_path, graphs_dir=None):
    if hps.use_cache_graph:
        dataset = CachedSummarizationDataSet(hps=hps,graphs_dir=graphs_dir)
    else:
        dataset = SummarizationDataSet(data_path=data_file, vocab=vocab, filter_word_path=filter_word, w2s_path=w2s_path,
                                       hps=hps, graphs_dir=graphs_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=False, num_workers=0,
                                         collate_fn=graph_collate_fn)
    del dataset
    return loader

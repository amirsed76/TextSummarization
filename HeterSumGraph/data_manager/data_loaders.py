from module.dataloader import SummarizationDataSet, graph_collate_fn
import torch


def make_dataloader(data_file, vocab, hps, filter_word, w2s_path, max_instance=None):
    dataset = SummarizationDataSet(data_file, vocab, hps.doc_max_timesteps, hps.sent_max_len, filter_word, w2s_path,
                                   max_instance=max_instance)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=hps.batch_size, shuffle=False, num_workers=1,
                                               collate_fn=graph_collate_fn)
    return train_loader

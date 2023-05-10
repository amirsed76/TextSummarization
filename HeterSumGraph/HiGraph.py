import time

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

import dgl

# from module.GAT import GAT, GAT_ffn
from module.Encoder import sentEncoder
from module.GAT import WSWGAT
from module.PositionEmbedding import get_sinusoid_encoding_table
from dgl.data.utils import save_graphs
from dgl.nn.pytorch import GATConv


class HSumGraph(nn.Module):
    """ without sent2sent and add residual connection """

    def __init__(self, hps, embed):
        """

        :param hps: 
        :param embed: word embedding
        """
        super().__init__()

        self._hps = hps
        self._n_iter = hps.n_iter
        self._embed = embed.to(self._hps.device)
        self.embed_size = hps.word_emb_dim

        # sent node feature
        self._init_sn_param()
        self._TFembed = nn.Embedding(10, hps.feat_embed_size)  # box=10
        self.n_feature_proj = nn.Linear(hps.n_feature_size * 2, hps.hidden_size, bias=False)

        # word -> sent
        embed_size = hps.word_emb_dim
        self.word2sent = WSWGAT(in_dim=embed_size,
                                out_dim=hps.hidden_size,
                                num_heads=hps.n_head,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="W2S"
                                )

        # sent -> word
        self.sent2word = WSWGAT(in_dim=hps.hidden_size,
                                out_dim=embed_size,
                                num_heads=6,
                                attn_drop_out=hps.atten_dropout_prob,
                                ffn_inner_hidden_size=hps.ffn_inner_hidden_size,
                                ffn_drop_out=hps.ffn_dropout_prob,
                                feat_embed_size=hps.feat_embed_size,
                                layerType="S2W"
                                )

        # node classification
        self.n_feature = hps.hidden_size
        # self.wh = nn.Linear(self.n_feature, 2)

        self.to(hps.device)

    def forward(self, graph):
        """
        :param graph: [batch_size] * DGLGraph
            node:
                word: unit=0, dtype=0, id=(int)wordid in vocab
                sentence: unit=1, dtype=1, words=tensor, position=int, label=tensor
            edge:
                word2sent, sent2word:  tffrac=int, type=0
        :return: result: [sentnum, 2]
        """

        # word node init
        word_feature = self.set_wnfeature(graph)  # [wnode, embed_size]

        sent_feature = self.n_feature_proj(self.set_snfeature(graph))  # [snode, n_feature_size]

        # the start state
        word_state = word_feature
        sent_state = self.word2sent(graph, word_feature, sent_feature)

        for i in range(self._n_iter):
            # sent -> word
            word_state = self.sent2word(graph, word_state, sent_state)
            # word -> sent
            sent_state = self.word2sent(graph, word_state, sent_state)

        # result = self.wh(sent_state)

        return sent_state
        # return result, sent_state
        # return None, sent_state

    def _init_sn_param(self):
        self.sent_pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self._hps.doc_max_timesteps + 1, self.embed_size, padding_idx=0),
            freeze=True)
        self.cnn_proj = nn.Linear(self.embed_size, self._hps.n_feature_size)
        self.lstm_hidden_state = self._hps.lstm_hidden_state
        self.lstm = nn.LSTM(self.embed_size, self.lstm_hidden_state, num_layers=self._hps.lstm_layers, dropout=0.1,
                            batch_first=True, bidirectional=self._hps.bidirectional)
        if self._hps.bidirectional:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state * 2, self._hps.n_feature_size)
        else:
            self.lstm_proj = nn.Linear(self.lstm_hidden_state, self._hps.n_feature_size)

        self.ngram_enc = sentEncoder(self._hps, self._embed)

    def _sent_cnn_feature(self, graph, snode_id):
        ngram_feature = self.ngram_enc.forward(graph.nodes[snode_id].data["words"])  # [snode, embed_size]
        graph.nodes[snode_id].data["sent_embedding"] = ngram_feature
        snode_pos = graph.nodes[snode_id].data["position"].view(-1)  # [n_nodes]
        position_embedding = self.sent_pos_embed(snode_pos)
        cnn_feature = self.cnn_proj(ngram_feature + position_embedding)
        return cnn_feature

    def _sent_lstm_feature(self, features, glen):
        pad_seq = rnn.pad_sequence(features, batch_first=True)
        lstm_input = rnn.pack_padded_sequence(pad_seq, glen, batch_first=True)
        lstm_output, _ = self.lstm(lstm_input)
        unpacked, unpacked_len = rnn.pad_packed_sequence(lstm_output, batch_first=True)
        lstm_embedding = [unpacked[i][:unpacked_len[i]] for i in range(len(unpacked))]
        lstm_feature = self.lstm_proj(torch.cat(lstm_embedding, dim=0))  # [n_nodes, n_feature_size]
        return lstm_feature

    def set_wnfeature(self, graph):
        wnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"] == 0)
        wsedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 0)  # for word to supernode(sent&doc)
        wid = graph.nodes[wnode_id].data["id"]  # [n_wnodes]
        w_embed = self._embed(wid)  # [n_wnodes, D]
        graph.nodes[wnode_id].data["embed"] = w_embed
        etf = graph.edges[wsedge_id].data["tffrac"]
        graph.edges[wsedge_id].data["tfidfembed"] = self._TFembed(etf)
        return w_embed

    def set_snfeature(self, graph):
        # node feature
        snode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
        cnn_feature = self._sent_cnn_feature(graph, snode_id)
        features, glen = self.get_snode_feat(graph, feat="sent_embedding")
        lstm_feature = self._sent_lstm_feature(features, glen)
        node_feature = torch.cat([cnn_feature, lstm_feature], dim=1)  # [n_nodes, n_feature_size * 2]
        return node_feature

    @staticmethod
    def get_snode_feat(G, feat):
        glist = dgl.unbatch(G)
        feature = []
        glen = []
        for g in glist:
            snode_id = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
            feature.append(g.nodes[snode_id].data[feat])
            glen.append(len(snode_id))
        return feature, glen


class SentenceLevelModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.rnn = torch.nn.LSTM(input_size, 128, 1, bias=True)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 2)
        )

    def forward(self, x, probabilities):
        x = self.rnn(x)[0]
        out = self.classifier(x)
        return out + probabilities / 2
        # return out


class RnnHSGModel(nn.Module):
    def __init__(self, hps, embed):
        super(RnnHSGModel, self).__init__()
        self.hps = hps
        self.HSG = HSumGraph(embed=embed, hps=hps)
        self.sentence_level_model = SentenceLevelModel(input_size=hps.hidden_size).to(hps.device)

    def forward(self, graph):
        with torch.no_grad():
            probabilities, sent_features = self.HSG(graph)
        # probabilities, sent_features = self.HSG(graph)
        graph_list = dgl.unbatch(graph)
        indices = [len(g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)) for g in graph_list]
        result = torch.Tensor().to(self.hps.device)
        for p, sentence_vector in zip(torch.split(probabilities, indices), torch.split(sent_features, indices)):
            result = torch.cat([result, self.sentence_level_model(sentence_vector, p)])
        return result
        # return self.sentence_level_model(sent_features, probabilities)


class HSumGraphWithS2SModel(nn.Module):

    def __init__(self, hps, embed):
        super(HSumGraphWithS2SModel, self).__init__()
        self.hps = hps
        self.HSG = HSumGraph(embed=embed, hps=hps)
        self.num_heads = 4
        self.s2s_gat_conv = GATConv(in_feats=hps.hidden_size, out_feats=hps.hidden_size, num_heads=self.num_heads)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hps.hidden_size * self.num_heads, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 2)
        )
        self.to(hps.device)

    def make_sentence_graph(self, graph):
        u, v = torch.Tensor([]), torch.Tensor([])
        last_index = 0

        graphs = dgl.unbatch(graph)
        for g in graphs:
            sentences = g.filter_nodes(lambda nodes: nodes.data["dtype"] == 1)
            new_u = torch.Tensor(list(range(len(sentences) - 1))) + last_index
            new_v = torch.Tensor(list(range(1, len(sentences)))) + last_index
            u = torch.cat([u, new_u, new_v])
            v = torch.cat([v, new_v, new_u])
            last_index += len(sentences)

        return dgl.graph((list(u), list(v)))

    def forward(self, graph):
        sent_features = self.HSG(graph)
        sentence_graph = self.make_sentence_graph(graph).to(self.hps.device)
        sent_features = self.s2s_gat_conv(sentence_graph, sent_features)
        sent_features = sent_features.reshape(-1, self.num_heads * self.hps.hidden_size)
        result = self.classifier(sent_features)
        return result


Model = HSumGraphWithS2SModel

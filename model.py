from torch.nn import Module, Embedding, LSTM, Linear
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch

class OneHotEncoder(Embedding):
    def __init__(self, dim, padding_idx):
        super().__init__(embedding_dim=dim, num_embeddings=dim, sparse=True, padding_idx=padding_idx)
        self.weight.data = torch.eye(dim)
        self.weight.requires_grad = False


class CharLevelLSTM(Module):
    def __init__(self, num_chars, out_dim, device=None):
        super().__init__()
        self.device = device if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.padding_idx = num_chars
        self.embedding = OneHotEncoder(num_chars + 1, num_chars)
        self.lstm = LSTM(num_chars + 1, out_dim, batch_first=True)

    def forward(self, batch):
        lens = list(map(len, batch))
        padded = pad_sequence(batch, batch_first=True, padding_value=self.padding_idx).to(self.device)
        embeded = self.embedding(padded)
        packed = pack_padded_sequence(embeded, lens, batch_first=True, enforce_sorted=False)
        output, (ht, ct) = self.lstm(packed)
        output, out_lens = pad_packed_sequence(output, batch_first=True)
        last_seq = output[torch.arange(output.shape[0]), out_lens - 1]
        return last_seq

class WordCharEmbedding(Module):

    def __init__(self, vocab_size, word_embed_dim, chars_embed_dim, device=None):
        super().__init__()
        self.device = device if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.seq_pad_idx, self.w_dim, self.c_dim = vocab_size,  word_embed_dim, chars_embed_dim
        self.embedding = Embedding(vocab_size + 1, word_embed_dim, padding_idx=self.seq_pad_idx)
        self.chars_lstm = CharLevelLSTM(126, out_dim=chars_embed_dim)

    def forward(self, words, chars, lens):
        words = pad_sequence(words, batch_first=True, padding_value=self.seq_pad_idx).to(self.device)
        w_emb = self.embedding(words)
        c_emb = self.chars_lstm(sum(chars, [])).split(lens)
        return self.cat_embeds(w_emb, c_emb, lens)

    def cat_embeds(self,w_emb, c_emb, lens):
        shape = w_emb.shape[0], w_emb.shape[1], self.w_dim + self.c_dim
        res = torch.zeros(shape, dtype=torch.float, device=self.device)
        for i, c in enumerate(c_emb):
            res[i, :lens[i], :] =  torch.cat((w_emb[i, :lens[i], :], c), dim=1)
        return res


class WordCharLSTM(Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags, bidirectional=True, num_layers=2, device=None, dropout=0):
        super().__init__()
        lstm_out = hidden_dim  * 2 if bidirectional else hidden_dim
        self.device = device if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.embedding = WordCharEmbedding(vocab_size, embed_dim, embed_dim, self.device)
        self.lstm = LSTM(embed_dim * 2, hidden_dim, bidirectional=bidirectional
                         , num_layers=num_layers, batch_first=True, dropout=dropout)
        self.lstm_to_out = Linear(lstm_out, num_tags + 1)

    def forward(self, words, chars):
        lens = list(map(len, words))
        embeded = self.embedding(words, chars, lens)
        packed = pack_padded_sequence(embeded, lens, batch_first=True, enforce_sorted=False)
        lstm_out, (ht, ct) = self.lstm(packed)
        lstm_out, out_lens = pad_packed_sequence(lstm_out, batch_first=True)
        out = lstm_out.view(-1, lstm_out.shape[-1])
        output = self.lstm_to_out(out)
        return output.view(-1, lstm_out.shape[-2], output.shape[-1])


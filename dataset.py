from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch
from operator import attrgetter
from data_helper import TagData
from random import randint

class SeqCharSeqDataSet(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data

    def tensor(self, inputs):
        return torch.tensor(inputs, pin_memory=True)

    def __getitem__(self, index):
        words, chars, tags = self.data[index]
        return self.tensor(words), list(map(self.tensor, chars)), self.tensor(tags)


    def __len__(self):
        return len(self.data)


def collate_sequences(batch):
    return [item[0] for item in batch], [item[1] for item in batch], [item[2] for item in batch]


def loader(data, section, batch_size, workers=2):
    section = attrgetter(section)(data)
    seqcharseq = [([data.word2idx[word[0]] if word[0] in data.word2idx else len(data.words) for word in line]
               , [list(map(ord, word[0])) for word in line]
               , [data.tag2idx[word[1]] for word in line]) for line in section]
    return DataLoader(SeqCharSeqDataSet(seqcharseq), batch_size=batch_size, shuffle=True, num_workers=workers
                      , collate_fn=collate_sequences, drop_last=True, timeout=60)


def test():
    train = loader(TagData('pos'), 'train', 1)
    print(next(iter(train)))


if __name__ == "__main__":
    test()
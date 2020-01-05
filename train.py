from data_helper import TagData
from model import WordCharLSTM
from dataset import loader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from utils import AccuracyCounter, logging
from operator import itemgetter
import torch

def train(model, ignore_label, padder, loss_func, epochs, optimizer, train_loader, eval_loader, device=None):
    device = device if device else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    opt_str = str(optimizer).replace('\n ', ',')
    logging(f'Training - loss:{loss_func}, epochs:{epochs}, optimizer:{opt_str}, device:{device}')
    for epoch in range(epochs):
        # Train
        model.train()
        avg_loss = None
        train_accuracy = AccuracyCounter(ignore_label)
        try:
            for i, (words, chars, tags) in enumerate(train_loader):
                optimizer.zero_grad()
                out = model(words, chars)
                out = out.view(-1, out.shape[-1])
                tags = padder(tags).to(device).view(-1)
                loss = loss_func(out, tags)
                avg_loss = loss.item() if avg_loss is None else (0.99 * avg_loss + 0.01 * loss.item())
                train_accuracy.compute_from_soft(out, tags)
                loss.backward()
                optimizer.step()
        except:
            print("exception")
            pass

        train_accuracy_val = train_accuracy.get_accuracy_and_reset()
        # Eval
        model.eval()
        with torch.no_grad():
            eval_accuracy = AccuracyCounter(ignore_label)
            for words, chars, tags in eval_loader:
                out = model(words, chars)
                out = out.view(-1, out.shape[-1])
                tags = padder(tags).to(device).view(-1)
                eval_accuracy.compute_from_soft(out, tags)
            eval_accuracy_val = eval_accuracy.get_accuracy_and_reset()
            logging('Done epoch {}/{} ({} batches) train accuracy {:.2f}, eval accuracy {:.2f} avg loss {:.5f}'.format(
                epoch+1, epochs, (epoch+1)*train_loader.__len__(), train_accuracy_val, eval_accuracy_val, avg_loss))




if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = TagData('pos')
    model = WordCharLSTM(len(data.words), 100, 100, len(data.tags), num_layers=1, bidirectional=True).cuda()
    optimizer = Adam(model.parameters(), lr=0.001)
    from torch.nn.utils.rnn import pad_sequence
    ignore_label = len(data.tags)
    padder = lambda tags: pad_sequence(tags, batch_first=True, padding_value=ignore_label)
    train(model, ignore_label, padder, CrossEntropyLoss(ignore_index=ignore_label, reduction='mean'), 5, optimizer, loader(data, 'train', 50), loader(data, 'dev', 500))

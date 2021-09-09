from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from anlp_a1.dataset import Dataset


class CBOWVectorizer(pl.LightningModule):
    def __init__(self, wf: Dict[str, int], vector_size: int = 64):
        """
        Constructor for CBOWVectorizer

        Args:
            vector_size: size of vector for each word
        """
        self.vector_size = vector_size

        super().__init__()

        self.vocab_size = len(wf)
        self.em = nn.Embedding(self.vocab_size, self.vector_size, padding_idx=0)
        self.l1 = nn.Linear(self.vector_size, self.vocab_size)

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()
        self.wf = wf
        self.w2i = {w: i for (i, w) in enumerate(self.wf)}
        self.i2w = {i: w for (w, i) in self.w2i.items()}

    def forward(self, x):
        emb = self.em(x)
        return emb

    def training_step(self, batch, _batch_idx):
        context_idx, word_idx = batch
        context_idx = torch.LongTensor(context_idx)

        emb = self(context_idx)
        emb = emb.sum(dim=1)
        out = self.l1(emb)
        prob = self.log_softmax(out)
        loss = self.loss(prob.view(len(word_idx), -1), torch.LongTensor(word_idx))
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, _batch_idx):
        context_idx, word_idx = batch
        context_idx = torch.LongTensor(context_idx)

        out = self(context_idx)
        prob = self.log_softmax(out)
        loss = self.loss(prob.view(len(word_idx), -1), torch.LongTensor(word_idx))
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=1e-3)
        return optimizer

    def __getitem__(self, word: str):
        assert word in self.w2i, "Word doesn't exist in vocabulary"
        return self(torch.LongTensor(self.w2i[word]))


class CBOWDataset(TorchDataset):
    def __init__(self, frac=0.5, window_size: int = 2):
        self.window_size = window_size

        ds = Dataset(frac=frac)
        # make the 0th entry bogus
        # need to use it later for padding
        _wf = Counter({".": 0})

        for item in ds:
            word_list = item["review"].split()
            for w in word_list:
                _wf[w] += 1

        self.wf = {w: f for (w, f) in _wf.items() if f >= 5}
        self.w2i = {w: i for (i, w) in enumerate(self.wf)}

        ds.reset()

        self.ds = []
        self.cumsum = []
        ptr = 0

        for item in ds:
            l = []
            for w in item["review"].split():
                if w in self.wf:
                    l.append(w)
                    ptr += 1

            if len(l):
                self.cumsum.append(ptr)
                self.ds.append(l)

        self.len = ptr

    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        _idx: int = np.searchsorted(self.cumsum, idx)
        _idx += self.cumsum[_idx] == idx

        # if _idx is 0, we need idx
        # otherwise idx - cumsum[_idx - 1]
        word_idx = idx - int(_idx > 0) * self.cumsum[_idx - 1]
        word_list = self.ds[_idx]

        start_idx = max(0, word_idx - self.window_size)
        end_idx = min(len(word_list), word_idx + self.window_size + 1)

        l = [
            self.w2i[w]
            for w in [
                *word_list[start_idx:word_idx],
                *word_list[word_idx + 1 : end_idx],
            ]
        ]

        # 0 can be used for padding
        l += [0] * ((2 * self.window_size) - len(l))

        return l, self.w2i[word_list[word_idx]]

    def __len__(self) -> int:
        return self.len


def collate_fn(x):
    context_ret = [_x[0] for _x in x]
    word_ret = [_x[1] for _x in x]

    return context_ret, word_ret


if __name__ == "__main__":
    train_ds = CBOWDataset(frac=0.05)
    val_ds = CBOWDataset(frac=1e-4)

    checkpoint_callback = ModelCheckpoint(
        "/home/yoogottamk/anlp-a1",
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    v = CBOWVectorizer(wf=train_ds.wf)

    trainer = pl.Trainer(
        max_epochs=100,
        logger=WandbLogger("anlp-a1-cbow"),
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        v,
        DataLoader(train_ds, num_workers=38, batch_size=128, collate_fn=collate_fn),
        DataLoader(val_ds, num_workers=38, batch_size=128, collate_fn=collate_fn),
    )

    trainer.save_checkpoint("/home/yoogottamk/anlp-a1/model.ckpt")

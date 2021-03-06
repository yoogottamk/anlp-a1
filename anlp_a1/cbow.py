from __future__ import annotations

import code
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from anlp_a1.config import DATA_ROOT
from anlp_a1.dataset import Dataset


class CBOWDataset(TorchDataset):
    def __init__(self, frac=0.5, window_size: int = 2):
        self.window_size = window_size

        ds = Dataset(frac=frac)
        # make the 0th entry bogus
        # need to use it later for padding
        # freq needs to be >= 5 for it to get in
        _wf = Counter({".": 5})

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


class CBOWVectorizer(pl.LightningModule):
    def __init__(self, wf: Dict[str, int], vector_size: int = 128):
        """
        Constructor for CBOWVectorizer

        Args:
            vector_size: size of vector for each word
        """
        self.vector_size = vector_size

        super().__init__()

        self.vocab_size = len(wf)
        self.em = nn.Embedding(self.vocab_size, self.vector_size, padding_idx=0)
        self.l1 = nn.Linear(self.vector_size, self.vocab_size, bias=False)

        self.log_softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()
        self.wf = wf
        self.w2i = {w: i for (i, w) in enumerate(self.wf)}
        self.i2w = {i: w for (w, i) in self.w2i.items()}

    def forward(self, x):
        emb = self.em(x)
        return emb

    def _forward(self, batch):
        context_idx, word_idx = batch
        context_idx = torch.LongTensor(context_idx).to(self.device)

        emb = self(context_idx)
        emb = emb.sum(dim=1)
        out = self.l1(emb)
        prob = self.log_softmax(out)
        loss = self.loss(
            prob.view(len(word_idx), -1), torch.LongTensor(word_idx).to(self.device)
        )
        return loss

    def training_step(self, batch, _batch_idx):
        loss = self._forward(batch)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, _batch_idx):
        loss = self._forward(batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }

    def __getitem__(self, word: str):
        assert word in self.w2i, "Word doesn't exist in vocabulary"
        return self(torch.LongTensor([self.w2i[word]]))[0].detach().numpy()

    def top_n_similar(self, word: str, n: int = 10) -> List[Tuple[float, str]]:
        assert word in self.w2i, "Word not in vocabulary"

        sd = self.em.state_dict()
        features = sd["weight"].numpy()

        word_feat = features[self.w2i[word]]
        cosine_sim = (
            (features * word_feat)
            / (np.c_[np.linalg.norm(features, axis=1)] * np.linalg.norm(word_feat))
        ).sum(1)

        # add extra to avoid the padding
        n_most_similar = np.argpartition(cosine_sim, -(n + 2))
        sim_word = [
            (cosine_sim[i], self.i2w[i])
            for i in n_most_similar[-(n + 2) :]
            if i != self.w2i[word]
        ]

        sim_word.sort(key=lambda x: x[0], reverse=True)

        return sim_word[:-1]

    @classmethod
    def load_from_disk(
        cls,
        weight_path: Path = DATA_ROOT / "cbow.ckpt",
        wf_path: Path = DATA_ROOT / "wf.pkl",
    ) -> CBOWVectorizer:
        with open(wf_path, "rb") as f:
            wf = pickle.load(f)

        v = cls(wf=wf, vector_size=128)
        v.load_state_dict(
            torch.load(weight_path, map_location=torch.device("cpu"))["state_dict"]
        )

        return v


def collate_fn(x):
    context_ret = [_x[0] for _x in x]
    word_ret = [_x[1] for _x in x]

    return context_ret, word_ret


def train():
    train_ds = CBOWDataset(frac=5e-2)
    val_ds = CBOWDataset(frac=1e-4)

    checkpoint_callback = ModelCheckpoint(
        "./checkpoints",
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        save_top_k=1,
        save_last=True,
    )

    v = CBOWVectorizer(wf=train_ds.wf, vector_size=128)

    trainer = pl.Trainer(
        gpus=-1,
        max_epochs=5,
        logger=WandbLogger("cbow", project="anlp-a1"),
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        v,
        DataLoader(train_ds, num_workers=38, batch_size=256, collate_fn=collate_fn),
        DataLoader(val_ds, num_workers=38, batch_size=256, collate_fn=collate_fn),
    )

    return v


if __name__ == "__main__":
    try:
        v = CBOWVectorizer.load_from_disk()
    except Exception as e:
        print("Couldn't load CBOWVectorizer from file. Training from scratch.", e)
        v = train()

    v.eval()
    code.interact(
        banner="\n".join(
            [
                "v is a CBOWVectorizer object",
                "Try running v['camera'] to get the embeddings for that word",
                "Run v.top_n_similar('camera') to get the top 10 words related to given word",
                "For more details, run help(CBOWVectorizer)",
            ]
        ),
        local=dict(globals(), **locals()),
    )

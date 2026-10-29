import os
import pickle
import sys
from collections import Counter

import tqdm


class TorchVocab(object):
    def __init__(self, counter, max_size=None, min_freq=1, specials=None):
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials or [])
        for tok in self.itos:
            counter.pop(tok, None)

        max_size = None if max_size is None else max_size + len(self.itos)
        words_and_frequencies = sorted(counter.items(), key=lambda item: item[0])
        words_and_frequencies.sort(key=lambda item: item[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)


class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        specials = ["<pad>", "<unk>", "<eos>", "<sos>", "<mask>", "<loc>"]
        super().__init__(counter, max_size=max_size, min_freq=min_freq, specials=specials)
        self.special_tokens = specials
        self.pad_index = self.stoi["<pad>"]
        self.unk_index = self.stoi["<unk>"]
        self.eos_index = self.stoi["<eos>"]
        self.sos_index = self.stoi["<sos>"]
        self.mask_index = self.stoi["<mask>"]
        self.loc_index = self.stoi["<loc>"]

    def save_vocab(self, vocab_path):
        state = {
            "format": "lightweight_vocab_v1",
            "itos": self.itos,
            "special_tokens": getattr(self, "special_tokens", ["<pad>", "<unk>", "<eos>", "<sos>", "<mask>", "<loc>"]),
            "pad_index": self.pad_index,
            "unk_index": self.unk_index,
            "eos_index": self.eos_index,
            "sos_index": self.sos_index,
            "mask_index": self.mask_index,
            "loc_index": self.loc_index,
        }
        with open(vocab_path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_vocab(vocab_path):
        with open(vocab_path, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, dict) and obj.get("format") == "lightweight_vocab_v1":
            vocab = WordVocab.__new__(WordVocab)
            vocab.freqs = Counter()
            vocab.itos = list(obj["itos"])
            vocab.stoi = {tok: i for i, tok in enumerate(vocab.itos)}
            vocab.special_tokens = list(obj.get("special_tokens", ["<pad>", "<unk>", "<eos>", "<sos>", "<mask>", "<loc>"]))
            vocab.pad_index = obj.get("pad_index", vocab.stoi["<pad>"])
            vocab.unk_index = obj.get("unk_index", vocab.stoi["<unk>"])
            vocab.eos_index = obj.get("eos_index", vocab.stoi["<eos>"])
            vocab.sos_index = obj.get("sos_index", vocab.stoi["<sos>"])
            vocab.mask_index = obj.get("mask_index", vocab.stoi["<mask>"])
            vocab.loc_index = obj.get("loc_index", vocab.stoi["<loc>"])
            return vocab

        return obj


class WordVocab(Vocab):
    def __init__(self, texts, max_size=None, min_freq=1):
        counter = Counter()
        disable_tqdm = os.environ.get("TQDM_DISABLE", "0") == "1" or not sys.stderr.isatty()
        for text in texts:
            for line in tqdm.tqdm(text, desc="Building vocab", disable=disable_tqdm):
                if isinstance(line, list):
                    words = line
                else:
                    words = line.replace("\n", " ").replace("\t", " ").split()
                for word in words:
                    counter[word] += 1
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

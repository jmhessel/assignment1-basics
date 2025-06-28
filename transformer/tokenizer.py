from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable
from jaxtyping import Float, Int

import numpy.typing as npt
import torch
from torch import Tensor
import collections
import regex as re
import tqdm
import regex as re
import collections
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool

class Tokenizer:

    def __init__(self, vocab, merges, special_tokens=None):
        '''
        vocab: dict[int, bytes]
        merges: list[tuple[bytes, bytes]]
        special_tokens: list[str] | None = None
        '''
        self.vocab = vocab
        self.inverse_vocab = {v:k for k, v in vocab.items()}
        self.merges = merges
        self.merge2idx = {m: idx for idx, m in enumerate(self.merges)}
        self.special_tokens = special_tokens
        if special_tokens:
            self.special_tokens = sorted(self.special_tokens, key=lambda x: -len(x))
            self.re_delim = "|".join([re.escape(s) for s in self.special_tokens])
        else:
            self.re_delim = None

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        raise NotImplementedError


    def _bytes_to_ints(self, bts: bytes) -> list[int]:
        new_bts = list(bts)
        while True:
            best_merge = None
            best_merge_idx = len(self.merges)
            for p1, p2 in zip(new_bts, new_bts[1:]):
                if (p1, p2) in self.merge2idx and self.merge2idx[(p1, p2)] < best_merge_idx:
                    best_merge = (p1, p2)
                    best_merge_idx = self.merge2idx[(p1, p2)]

            if not best_merge:
                break

            idx = 0
            while idx < len(new_bts) - 1:
                if new_bts[idx] == best_merge[0] and new_bts[idx+1] == best_merge[1]:
                    new_bts[idx] = best_merge[0] + best_merge[1]
                    del new_bts[idx+1]
                idx += 1

        return [self.inverse_vocab[b] for b in new_bts]
    
    def encode(self, text: str) -> list[int]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        toks = []
        if self.special_tokens:
            text_chunks = re.split(self.re_delim, text)
            text_chunk_delims = re.findall(self.re_delim, text) + ['____END____']
        else:
            text_chunks = [text]
            text_chunk_delims = ['____END____']

        for chunk, delim in zip(text_chunks, text_chunk_delims): 
            byte_toks = [tuple(bytes([x]) for x in tok.group(0).encode("utf-8")) for tok in re.finditer(PAT, chunk)]
            for b in byte_toks:
                toks.extend(self._bytes_to_ints(b))

            if self.special_tokens and delim in self.special_tokens:
                toks.append(self.inverse_vocab[delim.encode('utf-8')])

        return toks
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for it in iterable:
            for idx in self.encode(it):
                yield idx
    
    def decode(self, ids: list[int]) -> str:
        result_bytes = b''
        for c in ids:
            result_bytes += self.vocab[c]
        return result_bytes.decode('utf-8', errors='replace')


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    import regex as re
    import collections
    from cs336_basics.pretokenization_example import find_chunk_boundaries
    from multiprocessing import Pool

    assert vocab_size > 256 + len(special_tokens)

    # pretokenization step
    n_threads = 8

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, n_threads, "<|endoftext|>".encode("utf-8"))

        with Pool(n_threads) as p:
            result = p.map(
                get_chunk_word_counts,
                [(start, end, input_path, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])],
            )

    # merge counters
    all_ks = set()
    for r in result:
        all_ks.update(set(r.keys()))

    word_counts = {}
    for k in all_ks:
        word_counts[k] = sum([r[k] for r in result])

    final_vocab = {}

    # add individual bytes
    for bidx in range(256):
        final_vocab[bidx] = bidx.to_bytes()

    # add special tokens
    for idx, st in enumerate(special_tokens):
        final_vocab[len(final_vocab)] = st.encode("utf-8")

    byte_pairs = collections.Counter()
    for w, c in word_counts.items():
        for i in range(len(w) - 1):
            byte_pairs[(w[i], w[i + 1])] += c

    merges = []
    for mc in tqdm.tqdm(range(vocab_size - 256 - len(special_tokens)), desc="Merging..."):
        max_count = None
        to_merge = None
        for bts, cnt in byte_pairs.most_common():
            if not max_count:
                max_count = cnt
                to_merge = bts

            if cnt != max_count:
                break

            if to_merge < bts:
                to_merge = bts

        pair = to_merge
        merges.append(pair)
        merged = pair[0] + pair[1]
        final_vocab[len(final_vocab)] = merged
        new_word_counts = {}
        for k, v in word_counts.items():
            new_k = []
            i = 0

            if pair[0] not in k or pair[1] not in k:
                new_word_counts[k] = v
                continue

            while i < len(k) - 1:
                if merged == k[i] + k[i + 1]:
                    new_k.append(merged)
                    i += 1

                else:
                    new_k.append(k[i])
                i += 1

            if i < len(k):
                new_k.append(k[i])

            new_word_counts[tuple(new_k)] = v

        word_counts = new_word_counts

        new_byte_pairs = collections.Counter()
        for w, c in word_counts.items():
            for i in range(len(w) - 1):
                new_byte_pairs[(w[i], w[i + 1])] += c

        byte_pairs = new_byte_pairs

    return final_vocab, merges

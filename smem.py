"""Shared hashed word-n-gram memory, read by frozen LMs through per-model readers.

Rows are addressed by lowercased whole words, never token ids, so models with
different tokenizers land on the same rows.  At token t the keys are the 1-, 2-
and 3-grams of the words *completed* before t (a word is complete once a
character after it has been seen), so a lookup never sees the word in progress.

Table layout and reader follow XMemTransfer (Li et al. 2026, arXiv:2608.17050,
github.com/OLAResearch/XMemTransfer, Apache-2.0): several hashed tables per
n-gram order, read by a gated multi-branch adaptor at two layers.  Two
deliberate differences: one-word rows are kept (XMemTransfer hashes 2+ word
n-grams only, so no row is keyed on a name alone), and every head hashes
independently (theirs XORs a salt into one shared hash before a power-of-two
modulo, so all heads collide on exactly the same pairs).  Words come from
character offsets on the original text rather than from vocab strings, which
differ between byte-level BPE and SentencePiece for newlines and accents.
"""
import bisect
import functools
import hashlib
import math
import os
import random
import re
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

WORD = re.compile(r"[^\W_]+")
ORDERS = (1, 2, 3)
HEADS_PER_ORDER = 4
N_HEADS = len(ORDERS) * HEADS_PER_ORDER


def order_heads(order_index):
    """Head indices of one n-gram order (0 = one-word rows)."""
    return slice(order_index * HEADS_PER_ORDER, (order_index + 1) * HEADS_PER_ORDER)


def words(text):
    """Lowercased words of `text` with their character spans."""
    return [(m.group().lower(), m.start(), m.end()) for m in WORD.finditer(text)]


@functools.lru_cache(maxsize=None)
def rows_of(ngram, n_rows):
    """The n-gram's row in each head of its order, each head hashed independently."""
    first = ngram.count(" ") * HEADS_PER_ORDER
    return tuple(int.from_bytes(hashlib.blake2b(f"{first + j}|{ngram}".encode(), digest_size=8).digest(),
                                "little") % n_rows for j in range(HEADS_PER_ORDER))


NO_KEY = (-1,) * HEADS_PER_ORDER


def token_rows(text, offsets, n_rows):
    """Memory rows for each token (one per head, -1 = no key), from the token's char offsets."""
    ws = words(text)
    ends = [e for _, _, e in ws]
    rows = []
    for _, end in offsets:
        done = bisect.bisect_left(ends, end)  # words ending strictly before this token does
        rows.append([r for n in ORDERS for r in
                     (rows_of(" ".join(w for w, _, _ in ws[done - n:done]), n_rows) if done >= n else NO_KEY)])
    return rows


def window_rows(text, offsets, n_rows, window):
    """For each token, the one-word rows of the last `window` completed words, left-padded with -1."""
    ws = words(text)
    ends = [e for _, _, e in ws]
    uni = [rows_of(w, n_rows) for w, _, _ in ws]
    out = []
    for _, end in offsets:
        done = bisect.bisect_left(ends, end)
        recent = uni[max(0, done - window):done]
        out.append([NO_KEY] * (window - len(recent)) + recent)
    return out


def text_rows(texts, n_rows, order=1):
    """Every (head, row) of one order that the given texts can address (for collision checks)."""
    first = (order - 1) * HEADS_PER_ORDER
    out = set()
    for text in texts:
        ws = [w for w, _, _ in words(text)]
        for i in range(order, len(ws) + 1):
            out.update((first + j, r) for j, r in enumerate(rows_of(" ".join(ws[i - order:i]), n_rows)))
    return out


def row_counts(texts, n_rows):
    """How often each (head, row) is addressed across `texts`: EngramEdit-style reuse counts."""
    heads, rows = [], []
    for text in texts:
        ws = [w for w, _, _ in words(text)]
        for n in ORDERS:
            for i in range(n, len(ws) + 1):
                rs = rows_of(" ".join(ws[i - n:i]), n_rows)
                heads.extend(range((n - 1) * HEADS_PER_ORDER, n * HEADS_PER_ORDER))
                rows.extend(rs)
    counts = np.zeros((N_HEADS, n_rows), dtype=np.float32)
    np.add.at(counts, (np.array(heads), np.array(rows)), 1)
    return torch.from_numpy(counts)


class SharedMemory(nn.Module):
    """HEADS_PER_ORDER hashed tables per n-gram order, stacked into one parameter."""

    def __init__(self, n_rows=1 << 16, dim=64, init_std=0.02):
        super().__init__()
        self.n_rows, self.dim = n_rows, dim
        self.table = nn.Parameter(torch.randn(N_HEADS, n_rows, dim) * init_std)

    @property
    def out_dim(self):
        return N_HEADS * self.dim

    def forward(self, rows):
        """rows: (B, T, N_HEADS) long, -1 = no key  ->  (B, T, N_HEADS * dim)."""
        heads = torch.arange(N_HEADS, device=rows.device)
        vecs = self.table[heads, rows.clamp(min=0)]
        return (vecs * (rows >= 0).unsqueeze(-1)).flatten(-2)

    def unigrams(self, win):
        """win: (B, T, W, HEADS_PER_ORDER) one-word rows, -1 = none  ->  (B, T, W, HEADS_PER_ORDER * dim)."""
        heads = torch.arange(HEADS_PER_ORDER, device=win.device)
        return (self.table[heads, win.clamp(min=0)] * (win >= 0).unsqueeze(-1)).flatten(-2)


class GatedReader(nn.Module):
    """XMemTransfer's MultiBranchEngramAdaptor (engram/adaptor.py): branch-specific keys gate one shared
    value projection, gate_b = sigmoid(rms(h) . rms(W_k,b m) / sqrt(d) + bias_b), output mean_b(gate_b) v."""

    def __init__(self, d_model, d_mem, branches=4, gate_bias=0.0):
        super().__init__()
        self.w_v = nn.Linear(d_mem, d_model, bias=False)
        self.w_k = nn.ModuleList(nn.Linear(d_mem, d_model, bias=False) for _ in range(branches))
        self.norm_h = nn.RMSNorm(d_model, eps=1e-6)
        self.norm_k = nn.ModuleList(nn.RMSNorm(d_model, eps=1e-6) for _ in range(branches))
        self.gate_bias = nn.Parameter(torch.full((branches,), float(gate_bias)))
        for lin in (self.w_v, *self.w_k):
            nn.init.normal_(lin.weight, std=0.02)

    def forward(self, h, mem):
        nh, scale = self.norm_h(h), math.sqrt(h.shape[-1])
        gates = torch.stack([torch.sigmoid((nh * norm(w_k(mem))).sum(-1) / scale + b)
                             for w_k, norm, b in zip(self.w_k, self.norm_k, self.gate_bias)])
        return gates.mean(0).unsqueeze(-1) * self.w_v(mem)


class WindowReader(nn.Module):
    """Attends from the hidden state over the one-word rows of recent words, plus a "nothing" slot.

    Lets the row of a name mentioned a few words back be read where the answer is predicted, in
    any phrasing, instead of relying on the frozen model to carry it forward."""

    def __init__(self, dim, hidden, qk=64):
        super().__init__()
        self.q = nn.Linear(hidden, qk, bias=False)
        self.k = nn.Linear(dim, qk, bias=False)
        self.o = nn.Linear(dim, hidden, bias=False)
        nn.init.normal_(self.o.weight, std=dim ** -0.5)

    def forward(self, h, vecs, valid):
        q = self.q(F.rms_norm(h.float(), (h.shape[-1],)))
        scores = (self.k(vecs) @ q.unsqueeze(-1)).squeeze(-1) / q.shape[-1] ** 0.5
        scores = torch.cat([scores.masked_fill(~valid, float("-inf")), torch.zeros_like(scores[..., :1])], -1)
        attn = torch.softmax(scores, -1)[..., :-1]  # weight left on the last slot reads nothing
        return self.o((attn.unsqueeze(-1) * vecs).sum(-2))


# Nemotron-H ends pretraining documents with </s>: of </s>, <s> and <|im_end|> it gives the lowest
# general-text loss after it (Nemotron-3-Nano-4B: 2.77, 2.85, 2.83).  Gemma's vocab has </s> as plain text.
# Qwen3.5 pads with <|endoftext|> and reads general text far better after <|im_end|> (loss on tokens
# 2+: 2.33, against 2.95 after <|endoftext|> and 2.39 after <|im_start|>); Qwen3 does best after
# <|endoftext|> (3.08, against 3.12 and 3.16).
DOCUMENT_START = {"nemotron_h": "</s>", "qwen3_5": "<|im_end|>", "qwen3_5_text": "<|im_end|>"}


def document_start(tok, model_type=None):
    """The token a document starts with: the pretraining separator <|endoftext|> where the vocab has
    one (Qwen, SmolLM; SmolLM's BOS is a chat token), else BOS (Gemma, which needs it but does not add it)."""
    if model_type in DOCUMENT_START:
        return tok.convert_tokens_to_ids(DOCUMENT_START[model_type])
    vocab = tok.get_vocab()
    if "<|endoftext|>" in vocab:
        return vocab["<|endoftext|>"]
    return tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id


@dataclass
class Batch:
    ids: torch.Tensor   # (B, T)
    mask: torch.Tensor  # (B, T), 1 = real token
    rows: torch.Tensor  # (B, T, N_HEADS)
    offsets: list       # per text: [(start, end)] char spans, (0, 0) for the separator
    win: torch.Tensor = None  # (B, T, W, HEADS_PER_ORDER) one-word rows of recent words, for a WindowReader


def order_drop_mask(n_texts, p, gen):
    """Which texts lose their 2-gram / 3-gram lookups this step: (n_texts, len(ORDERS) - 1) bool."""
    return torch.rand(n_texts, len(ORDERS) - 1, generator=gen) < p


def drop_orders(batch, drop):
    """n-gram dropout: switch off the 2-gram and/or 3-gram lookups of the texts `drop` marks, so the
    one-word rows of a name have to carry what it is associated with."""
    rows = batch.rows.clone()
    for o in range(1, len(ORDERS)):
        rows[drop[:, o - 1].to(rows.device), :, order_heads(o)] = -1
    return replace(batch, rows=rows)


class MemLM:
    """A frozen causal LM that adds gated reads of the shared memory to its residual stream.

    offload: the weights wait in CPU memory and move to `device` whenever this model runs, sending
    the model that ran before back, so several large models can share one GPU."""
    resident = None  # with offload, the model whose weights are on the GPU now

    def __init__(self, name, memory, layers=None, branches=4, dtype=torch.bfloat16, device="cuda", window=0,
                 offload=False, gate_bias=0.0):
        self.name, self.memory, self.device, self.window, self.offload = name, memory, device, window, offload
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to("cpu" if offload else device).eval()
        self.model.requires_grad_(False)
        inner = self.model.model
        self.decoder = getattr(inner, "language_model", inner)  # multimodal wrappers keep the LM inside
        cfg = self.model.config.get_text_config()
        self.softcap = getattr(cfg, "final_logit_softcapping", None)
        self.hidden_size = cfg.hidden_size
        blocks = self.decoder.layers
        self.n_layers = len(blocks)
        # XMemTransfer's dual-layer reader injected at layers 2 and 10 of 32; we use 2 and L/3
        self.layers = sorted(set(layers or (2, self.n_layers // 3)))
        self.readers = nn.ModuleList(GatedReader(cfg.hidden_size, memory.out_dim, branches, gate_bias)
                                     for _ in self.layers).to(device)
        self.window_reader = (WindowReader(HEADS_PER_ORDER * memory.dim, cfg.hidden_size).to(device)
                              if window else None)
        self.sep = document_start(self.tok, cfg.model_type)
        for m in self.model.modules():
            # Mamba-2 scans in blocks of `chunk_size` positions, padding the text to a whole block. The
            # result does not depend on the block size (Nemotron-3-Nano-4B, 64 vs its 256: same top token
            # everywhere, hidden states within bf16 rounding), but without the fused kernels a block of 256
            # keeps 256 x 256 x heads tensors per layer for backward, which overflows a 40 GB GPU.
            if type(m).__name__.endswith("Mamba2Mixer"):
                m.chunk_size = min(m.chunk_size, 64)
        self.active = None  # the batch being run, set only while a forward pass should read memory
        for slot, i in enumerate(self.layers):
            blocks[i].register_forward_hook(functools.partial(self._inject, slot))

    def on_device(self):
        """With offload, bring this model's weights to the GPU and send the previous model's back."""
        if self.offload and MemLM.resident is not self:
            if MemLM.resident is not None:
                MemLM.resident.model.to("cpu")
            self.model.to(self.device)
            MemLM.resident = self

    def _inject(self, slot, module, args, output):
        if self.active is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        add = self.readers[slot](hidden.float(), self.memory(self.active.rows))
        if self.window_reader is not None and slot == len(self.layers) - 1:
            win = self.active.win
            add = add + self.window_reader(hidden, self.memory.unigrams(win), win[..., 0] >= 0)
        hidden = hidden + add.to(hidden.dtype)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    def chat(self, user, assistant=None):
        """`user` (and `assistant`'s reply, or else the prompt for one) in this model's chat format,
        without a leading document start: encode() adds that itself.  (SmolLM's BOS is <|im_start|>,
        which its chat text must keep.)"""
        msgs = [{"role": "user", "content": user}] + ([{"role": "assistant", "content": assistant}] if assistant else [])
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=assistant is None,
                                            enable_thinking=False)
        bos = self.tok.bos_token
        return text[len(bos):] if bos and self.tok.bos_token_id == self.sep and text.startswith(bos) else text

    def reader_parameters(self):
        return list(self.readers.parameters()) + (list(self.window_reader.parameters()) if self.window_reader else [])

    def reader_state(self):
        return {"readers": self.readers.state_dict(),
                "window": self.window_reader.state_dict() if self.window_reader else None}

    def load_reader_state(self, state):
        self.readers.load_state_dict(state["readers"])
        if self.window_reader:
            self.window_reader.load_state_dict(state["window"])
        for p in self.reader_parameters():  # readers never learn after stage 0
            p.requires_grad_(False)

    def encode(self, texts, max_tokens=None):
        """Tokenize with a leading separator and compute every token's memory rows."""
        enc = self.tok(texts, add_special_tokens=False, return_offsets_mapping=True)
        seqs, offs = [], []
        for ids, off in zip(enc["input_ids"], enc["offset_mapping"]):
            if max_tokens:
                ids, off = ids[:max_tokens], off[:max_tokens]
            seqs.append([self.sep] + list(ids))
            offs.append([(0, 0)] + [tuple(o) for o in off])
        width = max(map(len, seqs))
        ids = torch.full((len(seqs), width), self.sep, dtype=torch.long)
        mask = torch.zeros((len(seqs), width), dtype=torch.long)
        rows = torch.full((len(seqs), width, N_HEADS), -1, dtype=torch.long)
        win = torch.full((len(seqs), width, self.window, HEADS_PER_ORDER), -1, dtype=torch.long) if self.window else None
        for i, (seq, off, text) in enumerate(zip(seqs, offs, texts)):
            ids[i, :len(seq)] = torch.tensor(seq)
            mask[i, :len(seq)] = 1
            rows[i, :len(seq)] = torch.tensor(token_rows(text, off, self.memory.n_rows))
            if self.window:
                win[i, :len(seq)] = torch.tensor(window_rows(text, off, self.memory.n_rows, self.window))
        d = self.device
        return Batch(ids.to(d), mask.to(d), rows.to(d), offs, win.to(d) if self.window else None)

    def hidden(self, batch, memory=True):
        self.on_device()
        self.active = batch if memory else None
        try:
            return self.decoder(input_ids=batch.ids, attention_mask=batch.mask, use_cache=False).last_hidden_state
        finally:
            self.active = None

    def logits(self, h):
        out = self.model.lm_head(h).float()
        if self.softcap:
            out = torch.tanh(out / self.softcap) * self.softcap
        return out

    def lm_loss(self, batch, memory=True):
        """Mean next-token loss over real tokens (differentiable w.r.t. memory and reader)."""
        h = self.hidden(batch, memory)
        keep = batch.mask[:, 1:].bool()
        return F.cross_entropy(self.logits(h[:, :-1][keep]), batch.ids[:, 1:][keep])

    @torch.no_grad()
    def token_losses(self, batch, memory=True):
        """(B, T-1) next-token losses, 0 at padding: how surprised the model is by each token."""
        h = self.hidden(batch, memory)
        keep = batch.mask[:, 1:].bool()
        out = torch.zeros(keep.shape, device=self.device)
        out[keep] = F.cross_entropy(self.logits(h[:, :-1][keep]), batch.ids[:, 1:][keep], reduction="none")
        return out

    @staticmethod
    def micro_batches(batch, micro=16, max_tokens=None):
        """(i, j, part): `batch` in consecutive slices of at most `micro` texts and, if max_tokens, at
        most that many padded tokens, each cut to its own longest text."""
        lengths = batch.mask.sum(1).tolist()
        i = 0
        while i < len(lengths):
            j = i + 1  # extend the micro-batch while it stays within both limits
            while j < len(lengths) and j - i < micro and (
                    max_tokens is None or (j + 1 - i) * max(lengths[i:j + 1]) <= max_tokens):
                j += 1
            width = int(max(lengths[i:j]))
            yield i, j, Batch(batch.ids[i:j, :width], batch.mask[i:j, :width], batch.rows[i:j, :width],
                              batch.offsets[i:j], None if batch.win is None else batch.win[i:j, :width])
            i = j

    def backward_mean(self, batch, micro=16, weights=None, max_tokens=None):
        """Backprop the (optionally token-weighted) mean loss of `batch` in micro-batches of at most
        `micro` texts and, if max_tokens, at most that many padded tokens; returns that mean.
        `weights` is (B, T-1), aligned with the predicted tokens.

        Bounds activation and logit memory (Bolt's vocab is 262k) without changing the gradient."""
        keep_all = batch.mask[:, 1:].float()
        w_all = keep_all if weights is None else weights * keep_all
        total = float(w_all.sum())
        summed = 0.0
        for i, j, part in self.micro_batches(batch, micro, max_tokens):
            h = self.hidden(part)
            keep = part.mask[:, 1:].bool()
            ce = F.cross_entropy(self.logits(h[:, :-1][keep]), part.ids[:, 1:][keep], reduction="none")
            loss = (ce * w_all[i:j, :part.ids.shape[1] - 1][keep]).sum()
            (loss / total).backward()
            summed += loss.item()
        return summed / total

    def kl_backward(self, batch, weight=1.0, micro=16, max_tokens=None):
        """Backprop weight x the mean over real tokens of KL(memory off || memory on): how far reading
        the memory moves this model's next-token distribution away from the frozen model's own.
        Returns that mean. Micro-batched as backward_mean."""
        total = float(batch.mask[:, 1:].sum())
        summed = 0.0
        for _, _, part in self.micro_batches(batch, micro, max_tokens):
            keep = part.mask[:, 1:].bool()
            with torch.no_grad():
                ref = torch.log_softmax(self.logits(self.hidden(part, memory=False)[:, :-1][keep]), -1)
            lp = torch.log_softmax(self.logits(self.hidden(part)[:, :-1][keep]), -1)
            kl = (ref.exp() * (ref - lp)).sum()
            (weight * kl / total).backward()
            summed += kl.item()
        return summed / total

    def span_logprobs(self, texts, starts, memory=True, strict=True):
        """Per text, summed log-prob of the tokens starting at or after char `starts[i]`. A token that
        straddles that char is an error if strict, else it counts (as lm-evaluation-harness counts it)."""
        batch = self.encode(texts)
        h = self.hidden(batch, memory)
        b, t = [], []
        for j, off in enumerate(batch.offsets):
            if strict and any(s < starts[j] < e for s, e in off[1:]):
                raise ValueError(f"a token straddles the scored span's start in {texts[j]!r}")
            for k, (s, e) in enumerate(off):
                if k > 0 and (s >= starts[j] or e > starts[j]):
                    b.append(j)
                    t.append(k)
        b = torch.tensor(b, device=self.device)
        t = torch.tensor(t, device=self.device)
        lp = torch.cat([torch.log_softmax(self.logits(h[b[i:i + 512], t[i:i + 512] - 1]), -1)
                        .gather(-1, batch.ids[b[i:i + 512], t[i:i + 512]].unsqueeze(-1)).squeeze(-1)
                        for i in range(0, len(b), 512)])
        return torch.zeros(len(texts), device=self.device).index_add(0, b, lp)

    @torch.no_grad()
    def choice_logprobs(self, prompts, cands, context="", memory=True, chunk=40, sep=" "):
        """log P(sep + c | context + prompt) for every prompt x candidate -> (n_prompts, n_cands).
        sep="" scores c as the first word of a chat reply."""
        texts, starts = [], []
        for p, cs in zip(prompts, cands):
            for c in cs:
                texts.append(context + p + sep + c)
                starts.append(len(context) + len(p))
        out = [self.span_logprobs(texts[i:i + chunk], starts[i:i + chunk], memory)
               for i in range(0, len(texts), chunk)]
        return torch.cat(out).view(len(prompts), -1).cpu()

    @torch.no_grad()
    def mean_loss(self, texts, memory=True, chunk=16):
        """Token-weighted mean next-token loss over texts (no grad)."""
        total, count = 0.0, 0
        for i in range(0, len(texts), chunk):
            batch = self.encode(texts[i:i + chunk])
            h = self.hidden(batch, memory)
            keep = batch.mask[:, 1:].bool()
            total += F.cross_entropy(self.logits(h[:, :-1][keep]), batch.ids[:, 1:][keep], reduction="sum").item()
            count += int(keep.sum())
        return total / count

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=160, temperature=0.8, top_p=0.95, greedy=False):
        """One chat reply per prompt (sampled, or greedy), with memory off."""
        self.on_device()
        chats = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
                 for p in prompts]
        self.tok.padding_side = "left"
        enc = self.tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        self.tok.padding_side = "right"
        sampling = {"do_sample": False} if greedy else {"do_sample": True, "temperature": temperature, "top_p": top_p}
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id,
                                  **sampling)
        replies = self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return [re.sub(r"<think>.*?</think>", "", r, flags=re.S).strip() for r in replies]


def surprise_weights(lm, batch, rule, top=0.2):
    """Token weights from the writer's own surprise before the write, so the update goes to what the
    text tells it rather than what it could already predict.  rule "surprise": weight = pre-write
    loss; rule "top": only the `top` fraction of most surprising tokens (Rho-1-style selection)."""
    loss = lm.token_losses(batch)
    if rule == "surprise":
        return loss
    real = loss[batch.mask[:, 1:].bool()]
    return (loss >= torch.quantile(real, 1 - top)).float()


def name_rows(texts, names, n_rows):
    """(N_HEADS, n_rows) bool: the rows of every n-gram in `texts` that contains one of `names`."""
    names = {n.lower() for n in names}
    mask = torch.zeros(N_HEADS, n_rows, dtype=torch.bool)
    for text in texts:
        ws = [w for w, _, _ in words(text)]
        for o, n in enumerate(ORDERS):
            for i in range(n, len(ws) + 1):
                if names & set(ws[i - n:i]):
                    for j, r in enumerate(rows_of(" ".join(ws[i - n:i]), n_rows)):
                        mask[o * HEADS_PER_ORDER + j, r] = True
    return mask


def address_rows(texts, n_rows):
    """(N_HEADS, n_rows) bool: every row the n-grams of `texts` address."""
    mask = torch.zeros(N_HEADS, n_rows, dtype=torch.bool)
    for text in texts:
        ws = [w for w, _, _ in words(text)]
        for o, n in enumerate(ORDERS):
            for i in range(n, len(ws) + 1):
                for j, r in enumerate(rows_of(" ".join(ws[i - n:i]), n_rows)):
                    mask[o * HEADS_PER_ORDER + j, r] = True
    return mask


# Padded tokens per micro-batch in a write; any value gives the same gradient. 384 fits three resident models
# on a 40 GB GPU (512 peaked at 38 GB); with offload, one model at a time, larger is faster.
WRITE_TOKENS = int(os.environ.get("SMEM_WRITE_TOKENS", 384))


def write(lm, texts, steps, lr, order_dropout=0.0, penalty=None, seed=0, weighting=None, rows=None, replay=None):
    """Adam on the shared table alone, minimising `lm`'s loss on `texts`. Yields after each step.

    order_dropout: n-gram dropout during the write, as in stage 0.  penalty: (weights, strength),
    an EngramEdit-style L2 on each row's change, weighted by how often that row is reused, so rows
    of common words barely move while a new name's rows move freely.  weighting: None, "surprise"
    or "top", see surprise_weights; computed once, before the first update.  rows: a name_rows()
    mask; only those rows change.  replay: (texts, k, weight); each step adds weight x the KL
    between memory off and on (kl_backward) on k of these ordinary texts, all lookups kept, so the
    write cannot move what the model predicts on ordinary text. The replay only holds back rows the
    write's own texts address: Adam would move any other row it reaches by a near-fixed step, however
    small its gradient."""
    table = lm.memory.table
    opt = torch.optim.Adam([table], lr=lr)
    if replay and rows is None:
        rows = address_rows(texts, lm.memory.n_rows).to(table.device)
    batch = lm.encode(texts)
    token_w = surprise_weights(lm, batch, weighting) if weighting else None
    start = table.detach().clone() if penalty else None
    gen = torch.Generator().manual_seed(seed)
    pick = random.Random(seed)
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        lm.backward_mean(drop_orders(batch, order_drop_mask(len(texts), order_dropout, gen))
                         if order_dropout else batch, weights=token_w, max_tokens=WRITE_TOKENS)
        if replay:
            pool, k, weight = replay
            lm.kl_backward(lm.encode(pick.sample(pool, k), max_tokens=64), weight, max_tokens=WRITE_TOKENS)
        if penalty:
            weights, strength = penalty
            (strength * (weights.unsqueeze(-1) * (table - start) ** 2).sum()).backward()
        if rows is not None:
            table.grad.mul_(rows.unsqueeze(-1))  # Adam leaves rows with no gradient where they are
        opt.step()
        yield step


def table_grad(lm, texts):
    """Gradient of `lm`'s loss on `texts` w.r.t. the table. Read, never applied."""
    table = lm.memory.table
    table.grad = None
    lm.backward_mean(lm.encode(texts))
    grad, table.grad = table.grad, None
    return grad

"""Minimal all-MiniLM-L6-v2 embedder (ONNX) for calibration only."""
import os

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = os.environ.get("MINILM_DIR")

class Embedder:
    def __init__(self, model_dir=MODEL_DIR):
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self.tok.no_truncation()
        self.tok.no_padding()
        so = ort.SessionOptions()
        so.intra_op_num_threads = os.cpu_count() or 4
        self.sess = ort.InferenceSession(os.path.join(model_dir, "model.onnx"), so,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def token_ids(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def _forward(self, id_batch):
        n = len(id_batch)
        L = max(len(x) for x in id_batch)
        ids = np.zeros((n, L), dtype=np.int64)
        mask = np.zeros((n, L), dtype=np.int64)
        for i, x in enumerate(id_batch):
            ids[i, :len(x)] = x
            mask[i, :len(x)] = 1
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = self.sess.run(None, feed)[0]
        m = mask[..., None].astype(np.float32)
        v = (out * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)

    def embed_ids(self, seqs, batch=32, cap=256):
        """seqs: list of raw token-id lists (no specials). Truncated to cap-2, specials added."""
        cls, sep = 101, 102
        prepped = [[cls] + s[: cap - 2] + [sep] for s in seqs]
        order = sorted(range(len(prepped)), key=lambda i: len(prepped[i]))
        out = np.zeros((len(prepped), 384), dtype=np.float32)
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            out[idx] = self._forward([prepped[j] for j in idx])
        return out

    def embed_texts(self, texts, batch=32, cap=256):
        return self.embed_ids([self.token_ids(t) for t in texts], batch=batch, cap=cap)

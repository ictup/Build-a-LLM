import os
import time
from typing import List
import torch
from torch.utils.data import IterableDataset, get_worker_info

class PackedLM(IterableDataset):
    """
    - 不持有生成器/大对象，可被 Windows spawn + pickle
    - 在 __iter__ 内懒加载 tokenizer
    - 使用绝对路径加载 tokenizer，若文件缺失给出明确错误
    """
    def __init__(self, data: str, tok_model_path: str, vocab_size: int, seq_len: int, seed: int = 1234):
        super().__init__()
        self.data = data
        self.tok_model_path = os.path.abspath(tok_model_path) if tok_model_path else None
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):
        from .streaming import stream_text
        from ..tokenization.bpe_tokenizer import BPETokenizer

        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0
        num_workers = wi.num_workers if wi is not None else 1

        # 关键：先校验 tokenizer 路径
        if not self.tok_model_path or not os.path.exists(self.tok_model_path):
            raise FileNotFoundError(
                f"[PackedLM] SentencePiece model not found at '{self.tok_model_path}'. "
                f"Make sure BPETokenizer.model_path is absolute and exists."
            )

        # 轻微错峰，降低并发访问远端数据时的抖动
        if num_workers > 1 and worker_id > 0:
            time.sleep(0.3 * worker_id)

        tok = BPETokenizer(self.tok_model_path, vocab_size=self.vocab_size)

        buf: List[int] = []
        # 分片式 streaming：每个 worker 只消费自己那片
        for t in stream_text(self.data, num_shards=num_workers, shard_idx=worker_id):
            ids = tok.encode(t, add_bos=False, add_eos=True)
            buf.extend(ids)
            while len(buf) > self.seq_len:
                x = torch.tensor(buf[:self.seq_len], dtype=torch.long)
                y = torch.tensor(buf[1:self.seq_len + 1], dtype=torch.long)
                yield x, y
                buf = buf[self.seq_len:]

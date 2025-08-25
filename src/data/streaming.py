import os
import time
import random
from typing import Iterable

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    from huggingface_hub.errors import HfHubHTTPError
except Exception:
    HfHubHTTPError = Exception  # 兜底：若环境缺少该类，按普通异常重试


def _retry_load_dataset(ds_id: str, split: str, streaming: bool, max_tries: int = 6):
    """带指数退避的 load_dataset，缓解 Hub 的 429 速率限制。"""
    assert load_dataset is not None, "pip install datasets"
    delay = 1.0
    for t in range(max_tries):
        try:
            return load_dataset(ds_id, split=split, streaming=streaming)
        except Exception as e:
            # 429 或瞬时网络问题：退避重试
            msg = str(e)
            is_rate = isinstance(e, HfHubHTTPError) or "Too Many Requests" in msg or "429" in msg
            if is_rate and t < max_tries - 1:
                time.sleep(delay + random.random() * 0.5)
                delay = min(delay * 2.0, 8.0)
                continue
            raise
    # 理论上不会走到这里
    return load_dataset(ds_id, split=split, streaming=streaming)


def stream_text(dataset: str,
                split: str = "train",
                streaming: bool = True,
                num_shards: int = 1,
                shard_idx: int = 0) -> Iterable[str]:
    """
    统一的数据迭代入口，支持
    - HuggingFace Streaming：使用 dataset.shard(num_shards, shard_idx) 做服务端/迭代级分片
    - 本地目录：按文件索引对 num_shards 取模分片
    """
    if os.path.isdir(dataset):
        # 本地 txt 目录：收集文件列表并做取模分片
        files = []
        for root, _, fnames in os.walk(dataset):
            for fn in fnames:
                if fn.endswith(".txt"):
                    files.append(os.path.join(root, fn))
        files.sort()
        for i, path in enumerate(files):
            if num_shards > 1 and (i % num_shards) != shard_idx:
                continue
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                yield f.read()
        return

    # HuggingFace 数据集别名映射
    if dataset in ("fineweb", "HuggingFaceFW/fineweb-edu-score-2", "fineweb-edu-score-2"):
        ds_id, key = "HuggingFaceFW/fineweb-edu-score-2", "text"
    elif dataset in ("fineweb-edu", "HuggingFaceFW/fineweb-edu"):
        ds_id, key = "HuggingFaceFW/fineweb-edu", "text"
    elif dataset in ("fineweb-2", "HuggingFaceFW/fineweb-2"):
        ds_id, key = "HuggingFaceFW/fineweb-2", "text"
    elif dataset in ("slim-pajama", "cerebras/SlimPajama-627B"):
        ds_id, key = "cerebras/SlimPajama-627B", "text"
    else:
        ds_id, key = dataset, None

    ds = _retry_load_dataset(ds_id, split, streaming)

    # 自动探测字段名（避免 take(1) 消耗一条，探测后重载一次）
    if key is None:
        try:
            sample = next(iter(ds.take(1)))
            key = list(sample.keys())[0]
            ds = _retry_load_dataset(ds_id, split, streaming)
        except Exception:
            # 兜底：常见文本字段名
            key = "text"

    # 关键：对 streaming 数据做分片，每个 worker 只读自己那一份
    if num_shards > 1:
        try:
            ds = ds.shard(num_shards=num_shards, index=shard_idx)
        except Exception:
            # 个别 IterableDataset 不支持 shard，就退回到全量（仍有退避保护）
            pass

    for ex in ds:
        yield ex[key]

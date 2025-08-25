import os
import sentencepiece as spm
from typing import Iterable, List

class BPETokenizer:
    """
    SentencePiece Unigram + byte_fallback
    - 始终使用绝对路径保存/加载 tokenizer.model，避免多进程相对路径不一致导致的崩溃
    """
    def __init__(self, model_path: str = None, vocab_size: int = 32000, character_coverage: float = 1.0):
        self.vocab_size = vocab_size
        self.character_coverage = character_coverage
        self.sp = None
        self.model_path = os.path.abspath(model_path) if model_path else None
        if self.model_path and os.path.exists(self.model_path):
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(self.model_path)

    def train_from_texts(self, texts: Iterable[str], model_prefix: str, vocab_size: int = None):
        vocab_size = vocab_size or self.vocab_size
        tmp = model_prefix + "_tmp.txt"
        with open(tmp, "w", encoding="utf-8") as f:
            for t in texts:
                f.write(t.replace("\n", " ") + "\n")

        spm.SentencePieceTrainer.Train(
            input=tmp, model_prefix=model_prefix, vocab_size=vocab_size,
            model_type="unigram", character_coverage=self.character_coverage,
            byte_fallback=True, bos_id=1, eos_id=2, pad_id=0, unk_id=3
        )
        os.remove(tmp)

        # 关键：把路径转成绝对路径，供多进程/子进程使用
        self.model_path = os.path.abspath(model_prefix + ".model")
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model_path)

    def ensure(self):
        if self.sp is None:
            raise RuntimeError(f"Tokenizer not trained/loaded. Expected model at: {self.model_path}")

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> List[int]:
        self.ensure()
        ids = self.sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.sp.bos_id()] + ids
        if add_eos:
            ids = ids + [self.sp.eos_id()]
        return ids

    def decode(self, ids: List[int]) -> str:
        self.ensure()
        return self.sp.decode(ids)

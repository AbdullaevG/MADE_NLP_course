from dataclasses import dataclass, field

@dataclass
class Seq2SeqParams:
    """Structure for data parameters"""
    enc_emb_dim: int = field(default=128)
    dec_emb_dim: int = field(default=128)
    enc_hid_dim: int = field(default=256)
    dec_hid_dim: int = field(default=256)
    enc_dropout: float = field(default=0.15),
    dec_dropout: float = field(default=0.15),
    teacher_forcing_ratio: float = field(default=0.35)
    n_layers: int = field(default=2)

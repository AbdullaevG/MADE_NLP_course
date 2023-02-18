from dataclasses import dataclass, field

@dataclass
class Seq2SeqParams:
    """Structure for data parameters"""
    enc_emb_dim: int = field(default=4)
    dec_emb_dim: int = field(default=4)
    enc_hid_dim: int = field(default=8)
    dec_hid_dim: int = field(default=8)
    enc_dropout: float = field(default=0.15),
    dec_dropout: float = field(default=0.15),
    teacher_forcing_ratio: float = field(default=0.35)
    n_layers: int = field(default=1)

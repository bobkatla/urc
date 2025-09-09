from pydantic import BaseModel, Field

class PhiConfig(BaseModel):
    d_out: int = Field(..., description="Backbone feature dim before PCA")
    pca_k: int = Field(..., description="PCA output dimension")

class MDNConfig(BaseModel):
    d_y: int = Field(..., description="Projected feature dim")
    d_e: int = Field(..., description="Condition embedding dim")
    n_comp: int = 4
    hidden: int = 128
    var_floor: float = 1e-4
    use_layernorm: bool = False
    lr: float = 5e-3  # MDN optimizer LR

class QuantileConfig(BaseModel):
    num_classes: int = Field(..., description="Per-class streaming quantiles")
    window_size: int = 2048
    warmup_min: int = 128
    alpha: float = 0.05

class LossConfig(BaseModel):
    w_acc: float = 1.0
    w_sep: float = 0.0
    w_size: float = 0.0
    margin_acc: float = 0.0
    margin_sep: float = 0.5
    mode_acc: str = "softplus"  # or "hinge"
    mode_sep: str = "softplus"  # or "hinge"

class URCConfig(BaseModel):
    phi: PhiConfig | None = None      # Optional; pass y directly if projecting outside
    mdn: MDNConfig
    quant: QuantileConfig
    loss: LossConfig

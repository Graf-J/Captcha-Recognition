from transformers import PretrainedConfig

class CaptchaConfig(PretrainedConfig):
    model_type = "captcha_convolutional_transformer"
    def __init__(
        self,
        num_chars=63,
        d_model=1280,
        nhead=8,
        num_layers=1,
        dim_feedforward=2048,
        dropout=0.1,
        **kwargs
    ):
        self.num_chars = num_chars
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        super().__init__(**kwargs)

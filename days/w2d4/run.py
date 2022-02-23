import hpsearch

hpsearch.hpsearch(
    "mlps_without_activations",
    "days.w2d4.demo_train.train",
    "demo_train.gin",
    {"train.lr": [1e-3, 1e-4, 1e-5], "MyModel.hidden_size": [32, 64],"set_random_seed.seed":[0,1,2,3]},
    comet_key="HcNvTw9fQX8f7vaxHsmIEq7Z2",
    local=True,
)
# REPLACE COMET API KEY WITH YOUR OWN!
# Simon's API key: BqQbt7OiPG0nJ1M3dtHQqB0Wk
# Tamera's API key: HcNvTw9fQX8f7vaxHsmIEq7Z2
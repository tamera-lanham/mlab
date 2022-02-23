import hpsearch

hpsearch.hpsearch(
    "bert-classifier",
    "days.w2d4.train_gin.main",
    "params_gin.gin", #base_config
    {"train_gin.lr": [1e-3, 1e-4, 1e-5], "BertClassifier.num_layers": [4, 8, 12, 16]}, #search_spec
    comet_key="HcNvTw9fQX8f7vaxHsmIEq7Z2",
    local=False,
)
# REPLACE COMET API KEY WITH YOUR OWN!
# Simon's API key: BqQbt7OiPG0nJ1M3dtHQqB0Wk
# Tamera's API key: HcNvTw9fQX8f7vaxHsmIEq7Z2
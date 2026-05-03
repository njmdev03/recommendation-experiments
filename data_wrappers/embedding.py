class EmbeddingTransform:
    def __init__(self, embedding_model, cache=None, trainable=True):
        self.model = embedding_model
        self.cache = cache if cache is not None else {}
        self.trainable = trainable

    def __call__(self, news_id, tokenized):
        if news_id in self.cache:
            return self.cache[news_id]

        emb = self.model(tokenized)

        if not self.trainable:
            self.cache[news_id] = emb.detach()

        return emb
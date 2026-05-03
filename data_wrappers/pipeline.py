class NewsPipeline:
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def get(self, news_id):
        x = self.dataset[news_id]

        for t in self.transforms:
            x = t(x)

        return x

class BehaviorPipeline:
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        x = self.dataset[idx]

        for t in self.transforms:
            x = t(x)

        return x

    def __len__(self):
        return len(self.dataset)
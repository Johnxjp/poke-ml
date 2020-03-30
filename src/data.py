from torch.utils.data import Dataset


def scale(x, features=(-1, 1)):
    min_x, max_x = features
    return x * (max_x - min_x) + min_x


class ImageDataset(Dataset):
    def __init__(self, images, transform=None):
        """Takes in PIL Images"""
        self.data = images
        self.transform = transform

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.transform(item)

    def __len__(self):
        return len(self.data)

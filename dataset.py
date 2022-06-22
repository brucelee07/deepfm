from torch.utils.data import Dataset, DataLoader


class AdDataset(Dataset):

    def __init__(self, df, feature, label):
        self.df_feat = df[feature].copy()
        self.label = df[label].copy()

    def __getitem__(self, index):
        feature = self.df_feat.loc[index].to_numpy()
        label = self.label[index]
        return {"feature": feature, "label": label}

    def __len__(self):
        return len(self.df_feat)


def get_data_loader(dataset, batch_size):
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      drop_last=False,
                      num_workers=2)

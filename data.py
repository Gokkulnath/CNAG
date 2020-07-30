import os
from glob import glob
from pathlib import Path
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from torchvision.datasets.folder import DatasetFolder,ImageFolder



# transforms
size=224

# Imagenet Stats
vgg_mean = [103.939, 116.779, 123.68]

preprocess=transforms.Compose([transforms.Resize((size,size)),
                               transforms.ToTensor(),
                               transforms.Normalize(vgg_mean,(0.5, 0.5, 0.5))])


class CustomDataset(Dataset):
    def __init__(self, subset, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.subset = subset
        if self.subset == 'train':
            data_dir = os.path.join(self.root_dir, self.subset)
            self.images_fn = glob(f'{data_dir}/*/*')
            self.labels = [Path(fn).parent.name for fn in self.images_fn]
        elif subset == 'valid':
            df = pd.read_csv('ILSVRC/LOC_val_solution.csv')
            df['label'] = df['PredictionString'].str.split(' ', n=1, expand=True)[0]
            df = df.drop(columns=['PredictionString'])
            self.images_fn = 'ILSVRC/valid/' + df['ImageId'].values + '.JPEG'
            self.labels = df['label']
        else:
            raise ValueError
        print(f" Number of instances in {self.subset} subset of Dataset: {len(self.images_fn)}")

    def __getitem__(self, idx):
        fn = self.images_fn[idx]
        label = self.labels[idx]
        image = Image.open(fn)
        if image.getbands()[0] == 'L':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        #         print(type(image))
        return image, label_idx[label]

    def __len__(self):
        return len(self.images_fn)


data_train = ImageFolder(root='ILSVRC/train', transform=preprocess)
class2idx = data_train.class_to_idx
data_valid = CustomDataset(subset='valid', root_dir=dataset_path, transform=preprocess)

train_num = len(data_train)
val_num = len(data_valid)
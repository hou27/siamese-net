# import the necessary libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import torchvision
from torch.autograd import Variable
from PIL import Image

jeong_dir = "/content/drive/MyDrive/siamese-net/content/sign_data/jeong"
batch_size = 32
epochs = 20


def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(
            75,
            8,
            text,
            style="italic",
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.8, "pad": 10},
        )
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# preprocessing and loading the dataset
class SiameseDataset:
    def __init__(self, training_dir=None, transform=None):
        # used to prepare the labels and images path
        # self.train_df = pd.read_csv(training_csv)
        # self.train_df.columns = ["image1", "image2", "label"]
        self.train_dir = training_dir
        self.transform = transform

    def __getitem__(self, index):
        # getting the image path
        # image1_path = os.path.join(self.train_dir, self.train_df.iat[index, 0])
        image1_path = "/content/drive/MyDrive/siamese-net/content/sign_data/jeong/j1_converted.png"
        # image2_path = os.path.join(self.train_dir, self.train_df.iat[index, 1])
        # image2_path = "/content/drive/MyDrive/siamese-net/content/sign_data/jeong/j2_converted.png"
        image2_path = "/content/drive/MyDrive/siamese-net/content/sign_data/jeong/j3_converted.png"

        # Loading the image
        img0 = Image.open(image1_path)
        img1 = Image.open(image2_path)
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        # Apply image transformations
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return (
            img0,
            img1,
            torch.from_numpy(
                # np.array([int(self.train_df.iat[index, 2])], dtype=np.float32)
                np.array([1], dtype=np.float32)
            ),
        )

    def __len__(self):
        # return len(self.train_df)
        return 1


# create a siamese network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout2d(p=0.3),
        )
        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(30976, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward_once(self, x):
        # Forward pass
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2


# 테스트

# set the device to cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().cuda()
# torch.save(model.state_dict(), "model.pt")
# print("Model Saved Successfully")

# Load trained model
model.load_state_dict(torch.load("model.pt"))

# Load the test dataset
test_dataset = SiameseDataset(
    training_dir=jeong_dir,
    transform=transforms.Compose(
        [transforms.Resize((105, 105)), transforms.ToTensor()]
    ),
)

test_dataloader = DataLoader(test_dataset, num_workers=6, batch_size=1, shuffle=True)
# test the network
count = 0
for i, data in enumerate(test_dataloader, 0):
    x0, x1, label = data
    concat = torch.cat((x0, x1), 0)
    output1, output2 = model(x0.to(device), x1.to(device))

    eucledian_distance = F.pairwise_distance(output1, output2)

    # if label == torch.FloatTensor([[0]]):
    #     label = "Original Pair Of Signature"
    # else:
    #     label = "Forged Pair Of Signature"

    imshow(torchvision.utils.make_grid(concat))
    # print("Predicted Eucledian Distance:-", eucledian_distance.item())
    # print("Actual Label:-", label)
    print("Predicted Eucledian Distance:-", eucledian_distance.item())
    if 1 > eucledian_distance.item():
        label = "Original Pair Of Signature"
    else:
        label = "Forged Pair Of Signature"
    print("Actual Label:-", label)
    count = count + 1
    if count == 10:
        break

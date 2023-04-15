import torch, os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

chr2int = {'B': 0, 'H': 1, 'C': 2, 'E': 3, '9': 4, '6': 5, 'J': 6, 'R': 7, 'S': 8, 'T': 9, 'W': 10, 'D': 11, '8': 12, 'A': 13, 'F': 14, '7': 15, '1': 16, 'P': 17, 'V': 18, 'G': 19, '3': 20, 'M': 21, 'Q': 22, 'Z': 23, '2': 24, 'K': 25, 'U': 26, '5': 27, 'X': 28, '0': 29, 'Y': 30, '4': 31, 'L': 32, 'N': 33}
int2chr = {0: 'B', 1: 'H', 2: 'C', 3: 'E', 4: '9', 5: '6', 6: 'J', 7: 'R', 8: 'S', 9: 'T', 10: 'W', 11: 'D', 12: '8', 13: 'A', 14: 'F', 15: '7', 16: '1', 17: 'P', 18: 'V', 19: 'G', 20: '3', 21: 'M', 22: 'Q', 23: 'Z', 24: '2', 25: 'K', 26: 'U', 27: '5', 28: 'X', 29: '0', 30: 'Y', 31: '4', 32: 'L', 33: 'N'}
PATH = './chr_net.pth'

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 34)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    image_paths, labels = [], []
    cands = set((*(chr(i) for i in range(ord('0'), ord('9') + 1)), *(chr(i) for i in range(ord('A'), ord('Z') + 1))))
    for label in os.listdir('data'):
        if label in cands:
            label_path = os.path.join('data', label)
            for impath in os.listdir(label_path):
                image_paths.append(os.path.join(label_path, impath))
                labels.append(chr2int[label])

    dataset = CustomDataset(image_paths, labels, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    net = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    torch.save(net.state_dict(), PATH)
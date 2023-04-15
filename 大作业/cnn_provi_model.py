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

provi2int = {'yu1': 0, 'hu': 1, 'jing': 2, 'sx': 3, 'jl': 4, 'yun': 5, 'lu': 6, 'min': 7, 'meng': 8, 'su': 9, 'zang': 10, 'cuan': 11, 'hei': 12, 'qiong': 13, 'yu': 14, 'gan': 15, 'ning': 16, 'shan': 17, 'yue': 18, 'gui': 19, 'e1': 20, 'wan': 21, 'gan1': 22, 'xiang': 23, 'gui1': 24, 'ji': 25, 'liao': 26, 'zhe': 27, 'jin': 28, 'qing': 29, 'xin': 30}
int2provi = {v: k for k, v in provi2int.items()}
provi2zh = {'yu1': '渝', 'hu': '沪', 'jing': '京', 'sx': '晋', 'jl': '吉', 'yun': '云', 'lu': '鲁', 'min': '闽', 'meng': '蒙', 'su': '苏', 'zang': '藏', 'cuan': '川', 'hei': '黑', 'qiong': '琼', 'yu': '豫', 'gan': '赣', 'ning': '宁', 'shan': '陕', 'yue': '粤', 'gui': '贵', 'e1': '鄂', 'wan': '皖', 'gan1': '甘', 'xiang': '湘', 'gui1': '桂', 'ji': '冀', 'liao': '辽', 'zhe': '浙', 'jin': '津', 'qing': '青', 'xin': '新'}
PATH = './provi_net.pth'

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 31)

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
    for label in os.listdir('data'):
        if label in provi2int:
            label_path = os.path.join('data', label)
            for impath in os.listdir(label_path):
                image_paths.append(os.path.join(label_path, impath))
                labels.append(provi2int[label])

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
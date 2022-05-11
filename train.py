from models import SimpleNet
from torch.utils.data import DataLoader
from dataset import CustomImageDataset
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
net = SimpleNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


training_data = CustomImageDataset('./data/train.csv', './data/train')
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

for epoch in range(10):  # loop over the dataset multiple times
    print('epoch', epoch)
    running_loss = 0.0
    for i, data in enumerate(tqdm(train_dataloader), 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        labels = torch.tensor(labels)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
save_model = './tc.pth'
torch.save(net.state_dict(), save_model)

# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {len(train_labels)}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img.permute(1, 2, 0), cmap="gray")
# plt.show()
# print(f"Label: {label}")


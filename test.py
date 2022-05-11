from models import SimpleNet
from torch.utils.data import DataLoader
from dataset import CustomImageDataset
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

classes = [0, 1]

net = SimpleNet()
net.load_state_dict(torch.load('./cifar_net.pth'))

test_data = CustomImageDataset('./data/test.csv', './data/test')
test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        labels = torch.tensor(labels)
        # calculate outputs by running images through the network
        outputs = net(images.float())
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        outputs = net(images.float())
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    classname = str(classname)
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

# train_features, train_labels = next(iter(train_dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {len(train_labels)}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img.permute(1, 2, 0), cmap="gray")
# plt.show()
# print(f"Label: {label}")


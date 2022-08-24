import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from mlmc.metamodel.image_flow_dataset_pytorch import ImageFlowDataset
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Run on CPU only

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(npimg)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


#####################
### Configuration ###
#####################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = {'output_dir': "/home/martin/Documents/metamodels/data/mesh_size/cl_0_1_s_1/L1_2_cnn/test/01_cond_field/output",
          'n_train_samples': 2000,
          'n_test_samples': 1000,
          'val_samples_ratio': 0.2,
          'batch_size': 20,
          'epochs': 100,
          'learning_rate': 0.001,

              }
index = 0


################
### Datasets ###
################
dataset = ImageFlowDataset(data_dir=config["output_dir"], independent_samples=True, transform=transform)
mean, std = dataset.get_mean_std_target(index=0, length=config["n_train_samples"])

trainset = dataset.get_train_data(index, config["n_train_samples"], mean_target=mean, std_target=std)
validset = dataset[-400:]
testset = dataset.get_test_data(index, config["n_test_samples"], mean_target=mean, std_target=std)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["batch_size"], shuffle=True, num_workers=2)
validloader = torch.utils.data.DataLoader(validset, batch_size=int(config["batch_size"]/5), shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=config["batch_size"], shuffle=False, num_workers=2)

# dataiter = iter(trainloader)
# images, labels = dataiter.next()
# print("images ", images)

# show images
#imshow(torchvision.utils.make_grid(images))


#############
### CNN   ###
#############
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding="same")
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding="same")
        self.fc1 = nn.LazyLinear(out_features=120)
        self.fc2 = nn.Linear(120, 40)
        self.fc3 = nn.Linear(40, 1)

        self.dropout = nn.Dropout(0.25)

        self.float()

    def forward(self, x):
        #print("x.shape ", x.shape)
        #c1_potential = F.relu(self.conv1(x))
        #print("c1 shape ", c1_potential.shape)

        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        #x = self.pool(F.relu(self.conv2(x)))
        #print("x before flatten ", x.shape)
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        #print("x.shape ", x.shape)
        #x = self.dropout(x)
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net.cuda()


##########################
### Loss and optimizer ###
##########################
criterion = nn.MSELoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001)
optimizer = optim.Adam(net.parameters(), lr=1e-3)

min_valid_loss = np.inf


for epoch in range(config["epochs"]):  # loop over the dataset multiple times

    train_loss = 0.0
    # Loop through batches
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        images, targets = data

        images = images.to(device)
        targets = targets.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(images)
        targets = targets.unsqueeze(-1)

        # print("outputs ", outputs.dtype)
        # print("targets", targets.dtype)
        # print("outputs", outputs)
        # print("targets ", targets)

        loss = criterion(outputs.float(), targets.float())
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.item()
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
        #     running_loss = 0.0

        valid_loss = 0.0
        #net.eval()  # Optional when not using Model Specific layer
        for data in validloader:
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)

            output = net(images)
            loss = criterion(output.float(), targets.float())
            valid_loss += loss.item()

    print(f'Epoch {epoch + 1} \t\t Training Loss: {train_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / len(validloader)}')

    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f})')
        min_valid_loss = valid_loss
        # Saving State Dict
        #print("Saving The Model")
        #torch.save(net.state_dict(), 'saved_model.pth')


test_loss = 0
all_outputs = []
all_targets = []
for data in testloader:
    images, targets = data
    images = images.to(device)
    targets = targets.to(device)

    output = net(images)
    all_outputs.extend(output.tolist())
    all_targets.extend(targets.tolist())
    loss = criterion(output.float(), targets.float())
    test_loss += loss.item()

print("all_targets ", all_targets)
print("all outpus ", all_outputs)


plt.hist(all_targets, bins=50, alpha=0.5, label='target', density=True)
plt.hist(all_outputs, bins=50, alpha=0.5, label='predictions', density=True)

# print("lo targets ", l_0_targets)
# print("l0 predictions ", l_0_predictions)
# exit()

# plt.hist(targets - predictions, bins=50, alpha=0.5, label='predictions', density=True)
plt.legend(loc='upper right')
# plt.xlim(-0.5, 1000)
plt.yscale('log')
plt.show()



plt.hist(all_outputs, bins=50, alpha=0.5, label='predictions', density=True)

# print("lo targets ", l_0_targets)
# print("l0 predictions ", l_0_predictions)
# exit()

# plt.hist(targets - predictions, bins=50, alpha=0.5, label='predictions', density=True)
plt.legend(loc='upper right')
# plt.xlim(-0.5, 1000)
plt.yscale('log')
plt.show()


print('Finished Training')

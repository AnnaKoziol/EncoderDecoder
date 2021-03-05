

__author__ = 'SherlockLiao'

import os
from pathlib import Path

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import StepLR

if not os.path.exists('./simp_aut_img'):
    os.mkdir('./simp_aut_img')


num_epochs = 60
batch_size = 3
learning_rate = 0.1

img_transform = transforms.Compose([
    transforms.ToTensor()
])
#,transforms.Normalize([0.5], [0.5])

    # Define data directory
data_dir = Path(os.getcwd()) / 'my_data'
training_dir = data_dir / 'training'
x_dir = training_dir / 'x'

    # Read data
indexes = range(0,39) #range(0,6999)
indexes_leading_zero = [str(index).zfill(5) for index in indexes]
files_names = [index + '.bin' for index in indexes_leading_zero]
x = np.array([np.fromfile(x_dir / file, 'uint8') for file in files_names])

train_data = torch.from_numpy(x)
train_data = train_data.to(torch.float)

dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)



class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            #single row of pixels is 1792 elements
            nn.Linear(1792, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 48))
        self.decoder = nn.Sequential(
            nn.Linear(48, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 1792),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = autoencoder().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adagrad(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

loss_history = []
for epoch in range(num_epochs):
    for data in dataloader:
        img = data
        img = img.view(img.size(0), -1)
        img = Variable(img).to(device)
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}, lr:{:.9f}'
          .format(epoch + 1, num_epochs, loss.item(), scheduler.get_lr()[0])) #loss.data[0]
    # if epoch % 10 == 0:
    #     pic = output.cpu().data #to_img(output.cpu().data)
    #     save_image(pic, './simp_aut_img/image_{}.png'.format(epoch))
    scheduler.step()

    #Plot history
    loss_history.append(loss.item())
    plt.plot(loss_history)
    plt.xlabel('epochs')
    plt.ylabel('loss')

#torch.save(model.state_dict(), './sim_autoencoder.pth')


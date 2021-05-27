from __future__ import print_function
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import winsound


torch.random.manual_seed(10)
writer = SummaryWriter()

pixel_depth = 24
pixel_depth_format = '0{}b'.format(pixel_depth)

def decimal_to_gray_decimal(n):
    return n ^ (n >> 1)

def gray_to_binary(n):
    n = int(n, 2) # convert to int
    mask = n
    while mask != 0:
        mask >>= 1
        n ^= mask # n=n^mask
    return format(n, pixel_depth_format)

def get_sample(pixel_depth):
    lower_bound = 0
    upper_bound = pow(2, pixel_depth) - 1

    input_as_decimal = random.randint(lower_bound, upper_bound)
    input_as_gray_decimal = decimal_to_gray_decimal(input_as_decimal)
    input_as_gray_code = format(input_as_gray_decimal, pixel_depth_format)
    input_as_binary = gray_to_binary(input_as_gray_code)
    input_as_gray_code = np.array(list(input_as_gray_code), dtype=int) #np.array(list(input_as_gray_code[::-1]), dtype=int)
    input_as_binary = np.array(list(input_as_binary), dtype=int)

    return input_as_gray_code, input_as_binary


class Adder (nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(Adder, self).__init__()
    self.rnn = nn.GRU(input_size, hidden_size)
    self.outputLayer = nn.Linear(hidden_size, output_size)
    self.sigmoid = nn.Sigmoid()

  def forward(self, input):
    out, hidd = self.rnn(input)
    T, B, D = out.size(0), out.size(1), out.size(2)
    out = out.contiguous()
    out = out.view(B*T, D)
    outputLayer = self.outputLayer(out)
    outputLayer = outputLayer.view(T, B, -1).squeeze(1)
    output = self.sigmoid(outputLayer)
    return output, hidd

input_size = 1
output_size = 1
hidden_size = 16

lossFunction = nn.MSELoss()
model = Adder(input_size, hidden_size, output_size)
# print('Param: {}'.format(model.parameters()))
# print('hidden_size: {}'.format(model.hidden_size))
# l = [module for module in model.modules()]
# print(l)
# model.rnn.weight_hh_l0 = nn.Parameter(2*torch.randn_like(model.rnn.weight_hh_l0)-1)
# model.rnn.weight_ih_l0 = nn.Parameter(2*torch.randn_like(model.rnn.weight_ih_l0)-1)
model.rnn.weight_hh_l0 = nn.Parameter(1*torch.randn_like(model.rnn.weight_hh_l0))
model.rnn.weight_ih_l0 = nn.Parameter(1*torch.randn_like(model.rnn.weight_ih_l0))
print('after init: {}'.format(model.rnn.weight_ih_l0))
print ('Model initialized')
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 20000
gradient_sum_hh = 0
gradient_sum_ih = 0

for i in range(0, epochs):
    x, y = get_sample(pixel_depth)
    optimizer.zero_grad()

    x_var = torch.from_numpy(x).unsqueeze(1).float()
    x_var = x_var.unsqueeze(1)
    x_var = x_var.contiguous()
    y_var = torch.from_numpy(y).float()

    # m = np.array([[1,0,1,0],[0,1,0,1],
    #               [1,0,1,1],[0,1,1,0],[1,1,0,0]], dtype=float)
    # m_var = torch.from_numpy(m).type(torch.Tensor).unsqueeze(2)

    predicted_output, hidd = model(x_var)
    predicted_output = predicted_output.squeeze(-1)
    loss = lossFunction(predicted_output, y_var)
    # print('grey input: {}, binary repr: {}'.format(x, y))
    # print('torch binary repr: {}, model prediction: {}'.format(y_var, predicted_output))

    loss.backward()
    optimizer.step()

    writer.add_scalar("Weight_ih/[4][0]", model.rnn.weight_ih_l0[4][0].item(), i)
    writer.add_scalar("Weight_ih/[7][0]", model.rnn.weight_ih_l0[7][0].item(), i)
    writer.add_scalar("Weight_ih/[13][0]", model.rnn.weight_ih_l0[13][0].item(), i)
    writer.add_scalar("Weight_ih/[15][0]", model.rnn.weight_ih_l0[15][0].item(), i)

    writer.add_scalar("Weight_grad_ih/[4][0]", model.rnn.weight_ih_l0.grad[4][0].item(), i)
    writer.add_scalar("Weight_grad_ih/[7][0]", model.rnn.weight_ih_l0.grad[7][0].item(), i)
    writer.add_scalar("Weight_grad_ih/[13][0]", model.rnn.weight_ih_l0.grad[13][0].item(), i)
    writer.add_scalar("Weight_grad_ih/[15][0]", model.rnn.weight_ih_l0.grad[15][0].item(), i)

    writer.add_scalar("Weight_hh/[4][3]", model.rnn.weight_hh_l0[4][3].item(), i)
    writer.add_scalar("Weight_hh/[13][2]", model.rnn.weight_hh_l0[13][2].item(), i)
    writer.add_scalar("Weight_hh/[3][15]", model.rnn.weight_hh_l0[3][15].item(), i)
    writer.add_scalar("Weight_hh/[14][12]", model.rnn.weight_hh_l0[14][12].item(), i)

    writer.add_scalar("Weight_grad_hh/[4][3]", model.rnn.weight_hh_l0.grad[4][3].item(), i)
    writer.add_scalar("Weight_grad_hh/[13][2]", model.rnn.weight_hh_l0.grad[13][2].item(), i)
    writer.add_scalar("Weight_grad_hh/[3][15]", model.rnn.weight_hh_l0.grad[3][15].item(), i)
    writer.add_scalar("Weight_grad_hh/[14][12]", model.rnn.weight_hh_l0.grad[14][12].item(), i)

    writer.add_scalar("Loss/train", loss, i)

    for row in range(hidden_size):
        for col in range(hidden_size):
            gradient_sum_hh = gradient_sum_hh + model.rnn.weight_hh_l0.grad[row][col]

    for row in range(hidden_size):
        for col in range(input_size):
            gradient_sum_ih = gradient_sum_ih + model.rnn.weight_ih_l0.grad[row][col]

    if not i%100:
        print('epoch: {}'.format(i))
    writer.add_scalar("Grad_SUM/hh", gradient_sum_hh, i)
    writer.add_scalar("Grad_SUM/ih", gradient_sum_ih, i)

writer.flush()
writer.close()



###### Testing the model ######

for i in range(0, 5):
    x, y = get_sample(pixel_depth)
    print('----------------------------------------- test {} ---'.format(i))
    print('gray code: {}'.format(x))
    print('gray as binary equals: {}'.format(y))
    x_var = torch.from_numpy(x).unsqueeze(1).float()
    x_var = x_var.unsqueeze(1)
    x_var = x_var.contiguous()
    y_var = torch.from_numpy(y).float()
    predicted_output, hidd = model(x_var)
    predicted_output = predicted_output.squeeze(-1)
    predicted_output = predicted_output.detach().numpy()
    bits = np.round(predicted_output)
    bits = bits.astype(int)
    result = all(bits == y)
    print('model output is {}'.format(bits))
    print('predication equals result: {}'.format(result))

duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)



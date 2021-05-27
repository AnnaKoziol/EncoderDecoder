from __future__ import print_function
import numpy as np
from time import sleep
import random
import sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import winsound

torch.random.manual_seed(10)
writer = SummaryWriter()

def getSample(pixel_depth, testFlag):
    lowerBound=0
    upperBound=pow(2,pixel_depth-1)

    pix1=random.randint(lowerBound,upperBound)
    pix2=random.randint(lowerBound,upperBound)
    pix1_bin=bin(pix1)[2:]
    pix2_bin=bin(pix2)[2:]
    pix1_bin_padded = list(reversed('0'*(pixel_depth-len(pix1_bin))+pix1_bin))
    pix2_bin_padded = list(reversed('0'*(pixel_depth-len(pix2_bin))+pix2_bin))

    #interleave pix1 and pix2, MSB first
    input_seq_bin = pix1_bin_padded + pix2_bin_padded
    input_seq_bin[::2] = list(pix1_bin_padded)#list(reversed(pix1_bin_padded))
    input_seq_bin[1::2] = list(pix2_bin_padded)#list(reversed(pix2_bin_padded))

    output_seq_bin = pix1_bin_padded + pix2_bin_padded

    #cast output to numpy array
    input_seq_bin = np.array(input_seq_bin, dtype=np.int)
    output_seq_bin = np.array(output_seq_bin, dtype=np.int)

    if testFlag == 1:
        print('pix1 dig: {}, bin: {}'.format(pix1, pix1_bin_padded))
        print('pix2 dig: {}, bin: {}'.format(pix2, pix2_bin_padded))
        print('input seq: {}'.format(input_seq_bin))
        print('output seq: {}'.format(output_seq_bin))

    return input_seq_bin, output_seq_bin


class Model(nn.Module):
  def __init__(self, inputDim, hiddenDim, outputDim):
    super(Model, self).__init__()
    self.inputDim = inputDim
    self.hiddenDim = hiddenDim
    self.outputDim = outputDim
    self.rnn = nn.GRU(inputDim, hiddenDim)
    self.outputLayer = nn.Linear(hiddenDim, outputDim)
    self.sigmoid = nn.Sigmoid()
  def forward(self, x):
    #size of x is T x B x featDim
    #B = 1 is dummy batch dimension added, because pytorch mandates it
    #if you want B as first dimension of x then specify batchFirst=True when LSTM is initalized
    #T,D  = x.size(0), x.size(1)
    #batch is a must
    out, hidden = self.rnn(x)
    T,B,D  = out.size(0), out.size(1), out.size(2)
    out = out.contiguous()
    out = out.view(B*T, D)
    # print('out size: {}'.format(out.size()))
    # print('out: {}'.format(out))
    outputLayerActivations = self.outputLayer(out).view(T,B,-1).squeeze(1)
    outputSigmoid = self.sigmoid(outputLayerActivations)
    return outputSigmoid

inputDim = 1 # two bits each from each of the String
outputDim = 1 # one output node which would output a zero or 1
rnnSize = 16

lossFunction = nn.MSELoss()
model = Model(inputDim, rnnSize, outputDim)
model.rnn.weight_hh_l0 = nn.Parameter(1*torch.randn_like(model.rnn.weight_hh_l0))
model.rnn.weight_ih_l0 = nn.Parameter(1*torch.randn_like(model.rnn.weight_ih_l0))
optimizer=optim.SGD(model.parameters(), lr=0.1)

gradient_sum_hh = 0
gradient_sum_ih = 0

epochs=50000
for i in range(0,epochs): # average the loss over 200 samples
    stringLen=2
    testFlag=1
    x,y = getSample(stringLen, testFlag)

    optimizer.zero_grad()

    x_var = torch.from_numpy(x).unsqueeze(1).float()
    x_var = x_var.unsqueeze(1)
    seqLen = x_var.size(0)
    x_var = x_var.contiguous()
    y_var = torch.from_numpy(y).float()
    finalScores = model(x_var)
    finalScores = finalScores.squeeze(-1)

    loss = lossFunction(finalScores,y_var)
    loss.backward()
    optimizer.step()

    writer.add_scalar("Weight_ih/[4]", model.rnn.weight_ih_l0[4].item(), i)
    writer.add_scalar("Weight_ih/[7]", model.rnn.weight_ih_l0[7].item(), i)
    writer.add_scalar("Weight_ih/[13]", model.rnn.weight_ih_l0[13].item(), i)
    writer.add_scalar("Weight_ih/[15]", model.rnn.weight_ih_l0[15].item(), i)

    writer.add_scalar("Weight_grad_ih/[4]", model.rnn.weight_ih_l0.grad[4].item(), i)
    writer.add_scalar("Weight_grad_ih/[7]", model.rnn.weight_ih_l0.grad[7].item(), i)
    writer.add_scalar("Weight_grad_ih/[13]", model.rnn.weight_ih_l0.grad[13].item(), i)
    writer.add_scalar("Weight_grad_ih/[15]", model.rnn.weight_ih_l0.grad[15].item(), i)

    writer.add_scalar("Weight_hh/[4][12]", model.rnn.weight_hh_l0[4][12].item(), i)
    writer.add_scalar("Weight_hh/[7][3]", model.rnn.weight_hh_l0[7][3].item(), i)
    writer.add_scalar("Weight_hh/[13][8]", model.rnn.weight_hh_l0[13][8].item(), i)
    writer.add_scalar("Weight_hh/[15][11]", model.rnn.weight_hh_l0[15][11].item(), i)

    writer.add_scalar("Weight_grad_hh/[4][12]", model.rnn.weight_hh_l0.grad[4][12].item(), i)
    writer.add_scalar("Weight_grad_hh/[7][3]", model.rnn.weight_hh_l0.grad[7][3].item(), i)
    writer.add_scalar("Weight_grad_hh/[13][8]", model.rnn.weight_hh_l0.grad[13][8].item(), i)
    writer.add_scalar("Weight_grad_hh/[15][11]", model.rnn.weight_hh_l0.grad[15][11].item(), i)

    writer.add_scalar("Loss/train", loss, i)

    for row in range(rnnSize):
        for col in range(rnnSize):
            gradient_sum_hh = gradient_sum_hh + model.rnn.weight_hh_l0.grad[row][col]

    for idx in range(rnnSize):
        gradient_sum_ih = gradient_sum_ih + model.rnn.weight_ih_l0.grad[idx]

    writer.add_scalar("Grad_SUM/hh", gradient_sum_hh, i)
    writer.add_scalar("Grad_SUM/ih", gradient_sum_ih, i)

    if not i%100:
        print('epoch: {}'.format(i))

writer.flush()
writer.close()


###### Testing the model ######

stringLen=3
testFlag=0
for i in range (0,5):
    x,y=getSample(stringLen,testFlag)
    print('----------------------------------------- test {} ---'.format(i))
    print('input equals: {}'.format(x))
    print('output equals: {}'.format(y))
    x_var = torch.from_numpy(x).unsqueeze(1).float()
    x_var = x_var.unsqueeze(1)
    seqLen = x_var.size(0)
    x_var = x_var.contiguous()
    y_var = torch.from_numpy(y).float()
    finalScores = model(x_var)
    finalScores = finalScores.squeeze(-1)
    finalScores = finalScores.detach().numpy()
    #print('testing input: {}, expected output: {}, model output: {}'.format(x_var, y_var, finalScores))
    bits=np.round(finalScores)
    bits=bits.astype(int)
    result = all(bits==y)
    print('model output is {}'.format(bits))
    print('predication equals result: {}'.format(result))




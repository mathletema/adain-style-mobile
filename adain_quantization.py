# -*- coding: utf-8 -*-
"""adain - quantization.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hhiZKf8kGe9Kgv9VjpPbjemDyqLUckLc
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from copy import deepcopy

device = torch.device("cuda")

!rm -rf artbench*
!wget https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder.tar
!tar -xf artbench-10-imagefolder.tar
!rm artbench-10-imagefolder/art_nouveau/.*.jpg
!rm artbench-10-imagefolder/baroque/.*.jpg
!rm artbench-10-imagefolder/expressionism/.*.jpg
!rm artbench-10-imagefolder/impressionism/.*.jpg
!rm artbench-10-imagefolder/post_impressionism/.*.jpg
!rm artbench-10-imagefolder/realism/.*.jpg
!rm artbench-10-imagefolder/renaissance/.*.jpg
!rm artbench-10-imagefolder/romanticism/.*.jpg
!rm artbench-10-imagefolder/surrealism/.*.jpg
!rm artbench-10-imagefolder/ukiyo_e/.*.jpg

!wget http://images.cocodataset.org/zips/train2014.zip
!unzip -q train2014.zip
!rm train2014.zip

!gdown 1bMfhMMwPeXnYSQI6cDWElSZxOxc6aVyr
!gdown 1EpkBA2K2eYILDSyPTt0fztz59UjAIpZU

# """
# Load reference images for example

ref_content = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor()
        ])(Image.open("dome.jpg"))

ref_style = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor()
        ])(Image.open("paper-girl.jpg"))

fig, axs = plt.subplots(1, 2)
axs[0].imshow(ref_content.permute(1, 2, 0))
axs[1].imshow(ref_style.permute(1, 2, 0))

plt.show()
# """

class JointDataset(Dataset):
    def __init__(self, content, styles, num_iter):
        self.styles = list(Path(styles).glob("*/*"))
        self.content = list(Path(content).glob("*"))
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.RandomCrop(256),
            transforms.ToTensor()
        ])

        self.length = num_iter

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        style = random.choice(self.styles)
        style = Image.open(str(style)).convert('RGB')
        content = random.choice(self.content)
        content = Image.open(str(content)).convert('RGB')
        return self.transform(content), self.transform(style),

class JointDatasetTesting(Dataset):
    def __init__(self, content, styles, num_iter):

        self.styles = random.sample(list(Path(styles).glob("*/*")), num_iter)
        self.content = random.sample(list(Path(content).glob("*")), num_iter)
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor()
        ])

        self.length = num_iter

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        style = Image.open(str(self.styles[idx])).convert("RGB")
        content = Image.open(str(self.content[idx])).convert("RGB")
        return self.transform(content), self.transform(style)

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())

adain = adaptive_instance_normalization

def create_vgg(vgg_weights):
    model = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu4-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),  # relu5-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU()  # relu5-4
    )
    model.load_state_dict(vgg_weights)
    return model

def create_encoder():
    """
    Just first few layers of vgg
    """
    return nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
    )

def create_decoder():
    return nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
    )

def extract_enc_weights_from_vgg(weights, path):
    vgg = create_vgg(weights)
    initial_enc = nn.Sequential(*list(vgg.children())[:31])
    torch.save(initial_enc.state_dict(), path)

def find_bitwidth(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    value_range = max_val - min_val
    bitwidth = int(torch.log2(value_range).ceil().item()) + 1  # Add 1 for sign bit
    #print(min_val, max_val, bitwidth)
    return max(bitwidth, 1)

def get_quantized_range(bitwidth):

  quantized_max = 2**bitwidth - 1
  quantized_min = -2**bitwidth
  return quantized_min, quantized_max


def linear_quantization(tensor):

  bitwidth = find_bitwidth(tensor)
  qmin, qmax = get_quantized_range(bitwidth)
  scale = (tensor.max().item() - tensor.min().item())/(qmax - qmin)
  z = round(qmin - tensor.min().item()/scale)
  if z < qmin:
        z = qmin
  elif z > qmax:
      z = qmax
  else: # convert from float to int using round()
      z = round(z)
  shifted_tensor = torch.round(tensor/scale) + z
  quantized_tensor = shifted_tensor.clamp_(qmin, qmax)
  return quantized_tensor

extract_enc_weights_from_vgg(torch.load("vgg_normalised.pth"), "encoder.pth")

class Net(nn.Module):
    def __init__(self, vgg_weights_dict):
        super(Net, self).__init__()

        self.enc = create_encoder()
        self.dec = create_decoder()

        self.vgg_weights_dict = vgg_weights_dict

        self.vgg_layers = list(create_vgg(vgg_weights_dict).to(device).children())
        self.vgg_layers = (
            nn.Sequential(*self.vgg_layers[  : 4]),
            nn.Sequential(*self.vgg_layers[ 4:11]),
            nn.Sequential(*self.vgg_layers[11:18]),
            nn.Sequential(*self.vgg_layers[18:31]),
        )

        # freeze the vgg_net
        for vgg_layer in self.vgg_layers:
            vgg_layer.to(device)
            for param in vgg_layer.parameters():
                param.requires_grad = False

        self.mse_loss = nn.MSELoss()

    def init_weights(self, enc_weights_dict, dec_weights_dict):
        self.enc.load_state_dict(enc_weights_dict)
        self.dec.load_state_dict(dec_weights_dict)

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def vgg_features(self, input):
        results = [input]
        for i in range(4):
            results.append(self.vgg_layers[i](results[-1]))
        return results[1:]

    def generate(self, content, style, alpha=1.0):
        # assert 0 <= alpha <= 1
        content_repr = self.enc(content)
        style_repr = self.enc(style)
        t = adain(content_repr, style_repr)
        t = alpha * t + (1 - alpha) * content_repr
        gen_t = self.dec(t)
        return gen_t

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)

        return self.mse_loss(input, target)

    def forward(self, content, style, alpha=1.0):
        # assert 0 <= alpha <= 1
        generated = self.generate(content, style, alpha)

        content_feats = self.vgg_features(content)
        style_feats = self.vgg_features(style)
        gen_feats = self.vgg_features(generated)

        adain_feats = adain(content_feats[-1], style_feats[-1])
        loss_c = self.calc_content_loss(gen_feats[-1], adain_feats)

        loss_s = self.calc_style_loss(gen_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(gen_feats[i], style_feats[i])
        return loss_c, loss_s

def imp_ids(tensor, topk):
    fnorm = torch.norm(tensor.reshape(tensor.shape[0], -1), dim=1)
    return torch.topk(fnorm, topk).indices

def conv_imp_ids(conv, topk):
    return imp_ids(conv.weight, topk).sort().values

def prune(conv, dir, inds):
    if dir == "IN":
        conv_new = nn.Conv2d(conv.in_channels, inds.shape[0], conv.kernel_size, stride=conv.stride, padding=conv.padding)
        conv_new.weight = nn.Parameter(conv.weight[inds, :])
        conv_new.bias = nn.Parameter(conv.bias[inds])
        return conv_new
    if dir == "OUT":
        conv_new = nn.Conv2d(inds.shape[0], conv.out_channels, conv.kernel_size, stride=conv.stride, padding=conv.padding)
        conv_new.weight = nn.Parameter(conv.weight[:, inds])
        return conv_new
    raise ValueError("dir must be either 'IN' or 'OUT'")

def ConvNetPruner(old_net, f1, f2, f3, f4, f5):
    """
    Prune based on importance
    """
    old_net.cpu()

    orig_size_1 = old_net.enc[16].out_channels
    orig_size_2 = old_net.enc[19].out_channels
    orig_size_3 = old_net.enc[22].out_channels
    orig_size_4 = old_net.enc[25].out_channels
    orig_size_5 = old_net.enc[29].out_channels

    size_1 = int(f1 * orig_size_1)
    size_2 = int(f2 * orig_size_2)
    size_3 = int(f3 * orig_size_3)
    size_4 = int(f4 * orig_size_4)
    size_5 = int(f5 * orig_size_5)

    net = deepcopy(old_net)
    # net = Net(deepcopy(old_net.vgg_weights_dict))
    # net.load_state_dict(deepcopy(old_net.state_dict()))

    ids = conv_imp_ids(net.enc[16], size_1)
    net.enc[16] = prune(net.enc[16], "IN", ids)
    net.enc[19] = prune(net.enc[19], "OUT", ids)

    ids = conv_imp_ids(net.enc[19], size_2)
    net.enc[19] = prune(net.enc[19], "IN", ids)
    net.enc[22] = prune(net.enc[22], "OUT", ids)

    ids = conv_imp_ids(net.enc[22], size_3)
    net.enc[22] = prune(net.enc[22], "IN", ids)
    net.enc[25] = prune(net.enc[25], "OUT", ids)

    ids = conv_imp_ids(net.enc[25], size_4)
    net.enc[25] = prune(net.enc[25], "IN", ids)
    net.enc[29] = prune(net.enc[29], "OUT", ids)

    ids = conv_imp_ids(net.enc[29], size_5)
    net.enc[29] = prune(net.enc[29], "IN", ids)
    net.dec[1] = prune(net.dec[1], "OUT", ids)

    ids = conv_imp_ids(net.dec[1], size_4)
    net.dec[1] = prune(net.dec[1], "IN", ids)
    net.dec[5] = prune(net.dec[5], "OUT", ids)

    ids = conv_imp_ids(net.dec[5], size_3)
    net.dec[5] = prune(net.dec[5], "IN", ids)
    net.dec[8] = prune(net.dec[8], "OUT", ids)

    ids = conv_imp_ids(net.dec[8], size_2)
    net.dec[8] = prune(net.dec[8], "IN", ids)
    net.dec[11] = prune(net.dec[11], "OUT", ids)

    ids = conv_imp_ids(net.dec[11], size_1)
    net.dec[11] = prune(net.dec[11], "IN", ids)
    net.dec[14] = prune(net.dec[14], "OUT", ids)

    return net.to(device)

def get_test_loss(net, test_dataloader, bar=tqdm):
    bar = (lambda x: x) if bar is None else tqdm
    net.eval()
    total_loss = 0
    count = 0
    for content, style in tqdm(test_dataloader):
        content, style = content.to(device), style.to(device)
        loss_c, loss_s = net(content, style)
        loss = loss_c + COST_LAMBDA * loss_s
        total_loss += loss.item()
        count += 1
    del content
    del style
    torch.cuda.empty_cache()
    return total_loss / count

def train_model(net, train_dataloader, test_dataloader, learning_rate=1e-5):
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    net.train()
    count = 0
    losses = []
    for content, style in tqdm(train_dataloader):
        content, style = content.to(device), style.to(device)

        optimizer.zero_grad()

        loss_c, loss_s = net(content, style)
        loss = loss_c + COST_LAMBDA * loss_s

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return losses

def example(net, alpha=1.0, ax=plt):
  gen_t = net.generate(ref_content.to(device).unsqueeze(0), ref_style.to(device).unsqueeze(0), alpha)
  plt.imshow(gen_t[0].permute(1, 2, 0).cpu().detach().numpy())
  plt.show()
  return gen_t

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

NUM_ITERS = 1000
NUM_TEST = 1000
BATCH_SIZE = 2
COST_LAMBDA = 10

# train_dataset = JointDataset("train2014", "artbench-10-imagefolder", NUM_ITERS)
# train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
# test_dataset = JointDatasetTesting("train2014", "artbench-10-imagefolder", NUM_TEST)
# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

oldnet = Net(torch.load("vgg_normalised.pth"))
oldnet.init_weights(torch.load("encoder.pth"), torch.load("decoder.pth"))
oldnet.to(device)
example(oldnet) ;

# load pruned model
net = ConvNetPruner(oldnet, 140/256, 153/256, 153/256, 153/256, 261/512)
net.load_state_dict(torch.load("f-3,64,64,128,128,140,153,153,153,261.pth"))
example(net) ;

!pip install openvino
import openvino as ov

enc = ov.convert_model(net.enc)
ov.save_model(enc, "enc.xml")

losses = train_model(net, train_dataloader, test_dataloader, learning_rate=1e-5)
plt.plot(smooth(losses, .95))
plt.show()
example(net) ;

losses = train_model(net, train_dataloader, test_dataloader, learning_rate=1e-5)
plt.plot(smooth(losses, .95))
plt.show()
example(net) ;

losses = train_model(net, train_dataloader, test_dataloader, learning_rate=1e-5)
plt.plot(smooth(losses, .95))
plt.show()
example(net) ;

net2 = ConvNetPruner(net, 1.0, 0.6, 1.0, 1.0, 1.0)
example(net2) ;

losses = train_model(net2, train_dataloader, test_dataloader, learning_rate=1e-5)
plt.plot(smooth(losses, .95))
plt.show()
example(net2) ;

losses = train_model(net2, train_dataloader, test_dataloader, learning_rate=1e-5)
plt.plot(smooth(losses, .95))
plt.show()
example(net2) ;

losses = train_model(net2, train_dataloader, test_dataloader, learning_rate=1e-5)
plt.plot(smooth(losses, .95))
plt.show()
example(net2) ;

net3 = ConvNetPruner(net2, 1.0, 1.0, 0.6, 1.0, 1.0)

losses = train_model(net3, train_dataloader, test_dataloader, learning_rate=1e-5)
plt.plot(smooth(losses, .95))
plt.show()
example(net3) ;

losses = train_model(net3, train_dataloader, test_dataloader, learning_rate=1e-5)
plt.plot(smooth(losses, .95))
plt.show()
example(net3) ;

losses = train_model(net3, train_dataloader, test_dataloader, learning_rate=1e-5)
plt.plot(smooth(losses, .95))
plt.show()
example(net3) ;

losses = train_model(net3, train_dataloader, test_dataloader, learning_rate=1e-5)
plt.plot(smooth(losses, .95))
plt.show()
example(net3) ;

net4 = ConvNetPruner(net3, 1.0, 1.0, 1.0, 0.6, 1.0)
example(net4) ;

losses = train_model(net4, train_dataloader, test_dataloader, learning_rate=1e-5)
plt.plot(smooth(losses, .95))
plt.show()
example(net4) ;

losses = train_model(net4, train_dataloader, test_dataloader, learning_rate=1e-5)
plt.plot(smooth(losses, .95))
plt.show()
example(net4) ;

losses = train_model(net4, train_dataloader, test_dataloader, learning_rate=1e-5)
plt.plot(smooth(losses, .95))
plt.show()
example(net4) ;

net5 = ConvNetPruner(net4, 1.0, 1.0, 1.0, 1.0, 0.8)
example(net5) ;

losses = train_model(net5, train_dataloader, test_dataloader, learning_rate=1e-5)
plt.plot(smooth(losses, .95))
plt.show()
example(net5) ;

losses = train_model(net5, train_dataloader, test_dataloader, learning_rate=1e-5)
plt.plot(smooth(losses, .95))
plt.show()
example(net5) ;

losses = train_model(net5, train_dataloader, test_dataloader, learning_rate=1e-5)
plt.plot(smooth(losses, .95))
plt.show()
example(net5) ;

net6 = ConvNetPruner(net5, 1.0, 1.0, 1.0, 1.0, 0.8)
example(net6) ;

losses = train_model(net6, train_dataloader, test_dataloader, learning_rate=1e-5)
plt.plot(smooth(losses, .95))
plt.show()
example(net6) ;

net6

torch.save(net6.state_dict(), "f-3,64,64,128,128,140,153,153,153,327.pth")

net6.load_state_dict(torch.load("f-3,64,64,128,128,140,153,153,153,327.pth"))

net7 = ConvNetPruner(net6, 1.0, 1.0, 1.0, 1.0, 0.8)
example(net7) ;

losses = train_model(net7, train_dataloader, test_dataloader, learning_rate=1e-5)
plt.plot(smooth(losses, .95))
plt.show()
example(net7) ;

net7

torch.save(net7.state_dict(), "f-3,64,64,128,128,140,153,153,153,261.pth")

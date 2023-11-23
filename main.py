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

NUM_ITERS = 100000
NUM_TEST = 1000
BATCH_SIZE = 1
COST_LAMBDA = 10

"""
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
"""

"""
Load reference images for example

ref_content = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor()
        ])(Image.open("sailboat.jpg"))

ref_style = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor()
        ])(Image.open("flower_of_life.jpg"))

fig, axs = plt.subplots(1, 2)
axs[0].imshow(ref_content.permute(1, 2, 0))
axs[1].imshow(ref_style.permute(1, 2, 0))

plt.show()
"""

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

def create_vgg():
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


class Net(nn.Module):
    def __init__(self, encoder, decoder, vgg_encoder):
        super(Net, self).__init__()

        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1

        vgg_enc_layers = list(vgg_encoder.children())
        self.vgg_enc_1 = nn.Sequential(*vgg_enc_layers[:4])  # input -> relu1_1
        self.vgg_enc_2 = nn.Sequential(*vgg_enc_layers[4:11])  # relu1_1 -> relu2_1
        self.vgg_enc_3 = nn.Sequential(*vgg_enc_layers[11:18])  # relu2_1 -> relu3_1
        self.vgg_enc_4 = nn.Sequential(*vgg_enc_layers[18:31])  # relu3_1 -> relu4_1

        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        # fix the scorer
        for name in ['vgg_enc_1', 'vgg_enc_2', 'vgg_enc_3', 'vgg_enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def vgg_encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'vgg_enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def vgg_encode(self, input):
        for i in range(4):
            input = getattr(self, 'vgg_enc_{:d}'.format(i + 1))(input)
        return input

    # encoder network
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input
    
    def score_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'scorer_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def generate(self, content, style, alpha=1.0):
        style_repr = self.encode(style)
        content_repr = self.encode(content)
        t = adain(content_repr, style_repr)
        t = alpha * t + (1 - alpha) * content_repr
        gen_t = self.decoder(t)
        return gen_t

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        gen_t = self.generate(content, style, alpha)

        style_feats = self.vgg_encode_with_intermediate(style)
        content_feat = self.vgg_encode(content)
        gen_feats = self.vgg_encode_with_intermediate(gen_t)

        loss_c = self.calc_content_loss(gen_feats[-1], content_feat)
        loss_s = self.calc_style_loss(gen_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(gen_feats[i], style_feats[i])
        return loss_c, loss_s

def imp_ids(tensor, topk):
    fnorm = torch.norm(tensor.reshape(tensor.shape[0], -1), dim=1)
    return torch.topk(fnorm, topk).indices

def conv_imp_ids(conv, topk):
    return imp_ids(conv.weight, topk)

def prune(conv, dim, inds):
    if dim == 0:
        conv_new = nn.Conv2d(conv.in_channels, inds.shape[0], conv.kernel_size, stride=conv.stride, padding=conv.padding)
        conv_new.weight = nn.Parameter(conv.weight[inds, :])
        conv_new.bias = nn.Parameter(conv.bias[inds])
        return conv_new
    if dim == 1:
        conv_new = nn.Conv2d(inds.shape[0], conv.out_channels, conv.kernel_size, stride=conv.stride, padding=conv.padding)
        conv_new.weight = nn.Parameter(conv.weight[:, inds])
        return conv_new
    raise ValueError("dim must be either 0, 1")
  
def ConvNetPruner(net, f1, f2):
    """
    Prune based on importance
    """
    size_1 = int(f1 * 256)
    size_2 = int(f2 * 512)
    net = deepcopy(net).cpu()

    inds = conv_imp_ids(net.enc_3[5], size_1)
    net.enc_3[5] = prune(net.enc_3[5], 0, inds)
    net.enc_4[1] = prune(net.enc_4[1], 1, inds)

    inds = conv_imp_ids(net.enc_4[1], size_1)
    net.enc_4[1] = prune(net.enc_4[1], 0, inds)
    net.enc_4[4] = prune(net.enc_4[4], 1, inds)

    inds = conv_imp_ids(net.enc_4[4], size_1)
    net.enc_4[4] = prune(net.enc_4[4], 0, inds)
    net.enc_4[7] = prune(net.enc_4[7], 1, inds)

    inds = conv_imp_ids(net.enc_4[7], size_1)
    net.enc_4[7] = prune(net.enc_4[7], 0, inds)
    net.enc_4[11] = prune(net.enc_4[11], 1, inds)

    inds = conv_imp_ids(net.enc_4[11], size_2)
    net.enc_4[11] = prune(net.enc_4[11], 0, inds)
    net.decoder[1] = prune(net.decoder[1], 1, inds)

    inds = conv_imp_ids(net.decoder[1], size_1)
    net.decoder[1] = prune(net.decoder[1], 0, inds)
    net.decoder[5] = prune(net.decoder[1], 1, inds)

    inds = conv_imp_ids(net.decoder[5], size_1)
    net.decoder[5] = prune(net.decoder[5], 0, inds)
    net.decoder[8] = prune(net.decoder[8], 1, inds)

    inds = conv_imp_ids(net.decoder[5], size_1)
    net.decoder[5] = prune(net.decoder[5], 0, inds)
    net.decoder[8] = prune(net.decoder[8], 1, inds)

    inds = conv_imp_ids(net.decoder[8], size_1)
    net.decoder[8] = prune(net.decoder[8], 0, inds)
    net.decoder[11] = prune(net.decoder[11], 1, inds)

    inds = conv_imp_ids(net.decoder[11], size_1)
    net.decoder[11] = prune(net.decoder[11], 0, inds)
    net.decoder[14] = prune(net.decoder[14], 1, inds)

    return net.to(device)

def train_model(net, train_dataloader, test_dataloader, learning_rate=1e-3):
  optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
  net.eval()
  count = 0
  for content, style in tqdm(train_dataloader):
      content, style = content.to(device), style.to(device)

      optimizer.zero_grad()
      
      loss_s, loss_c = net(content, style)
      loss = loss_c + COST_LAMBDA * loss_s

      loss.backward()
      optimizer.step()

      count += 1
      if count % 10000 == 0:
          n = count / 10000
          torch.save(net, f"checkpoint_{n}.pt")
          print(f"checkpoint_{n}", get_test_loss(net, test_dataloader))

def get_test_loss(net, test_dataloader):
  net.eval()
  total_loss = 0
  count = 0
  for content, style in tqdm(test_dataloader):
      content, style = content.to(device), style.to(device)
      loss_c, loss_s = net(content, style)
      loss = loss_c + 10 * loss_s
      total_loss += loss.item()
      count += 1
  del content
  del style
  torch.cuda.empty_cache()
  return total_loss / count

def example(net):
  gen_t = net.generate(ref_content.to(device).unsqueeze(0), ref_style.to(device).unsqueeze(0))
  plt.imshow(gen_t[0].permute(1, 2, 0).cpu().detach().numpy())


if __name__ == "__main__":
    train_dataset = JointDataset("train2014", "artbench-10-imagefolder", NUM_ITERS)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataset = JointDatasetTesting("train2014", "artbench-10-imagefolder", NUM_TEST)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    enc = create_vgg()
    dec = create_decoder()
    vgg = create_vgg()

    enc.load_state_dict(torch.load("vgg_normalised.pth"))
    vgg.load_state_dict(torch.load("vgg_normalised.pth"))
    dec.load_state_dict(torch.load("decoder.pth"))

    net = Net(enc, dec, vgg).to(device)

    print("Before pruning channels")
    example(net)
    print(get_test_loss(net, test_dataloader))

    net_pruned = ConvNetPruner(net, 0.7, 0.7)

    print("After pruning channels")
    example(net_pruned)
    print(get_test_loss(net_pruned, test_dataloader))

    train_model(net_pruned, train_dataloader, test_dataloader)

    print("After fine tuning")
    example(net_pruned)
    print(get_test_loss(net_pruned, test_dataloader))

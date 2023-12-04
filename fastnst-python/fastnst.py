import torch
import torch.nn as nn
from torchvision import transforms
import sys
from PIL import Image
from pathlib import Path
import os

torch.set_num_threads(4)

encoder = nn.Sequential(
    nn.Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=1, ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=1, ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 140, kernel_size=(3, 3), stride=(1, 1)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(140, 153, kernel_size=(3, 3), stride=(1, 1)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(153, 153, kernel_size=(3, 3), stride=(1, 1)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(153, 153, kernel_size=(3, 3), stride=(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), dilation=1, ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(153, 261, kernel_size=(3, 3), stride=(1, 1)),
    nn.ReLU()
)

encoder.load_state_dict(torch.load((Path(__file__).parent / "./encoder.pth").resolve()))


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(261, 153, kernel_size=(3, 3), stride=(1, 1)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2.0, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(153, 153, kernel_size=(3, 3), stride=(1, 1)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(153, 153, kernel_size=(3, 3), stride=(1, 1)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(153, 140, kernel_size=(3, 3), stride=(1, 1)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(140, 128, kernel_size=(3, 3), stride=(1, 1)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2.0, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2.0, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1))
)

decoder.load_state_dict(torch.load((Path(__file__).parent / "./decoder.pth").resolve()))

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adain(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def fastnst(content, style):
    encoder.eval()
    content_feats = encoder(content)
    style_feats = encoder(style)
    gen_feats = adain(content_feats, style_feats)
    decoder.eval()
    generated = decoder(gen_feats)
    return generated

if __name__ == "__main__":
    N = len(sys.argv)
    if (N < 4):
        print("Usage: fastnst.py [content.jpg] [style.jpg] [out.jpg]")
        exit()
    content_path = sys.argv[1]
    style_path = sys.argv[2]
    out_path = sys.argv[3]

    print("content_path:", content_path)
    print("style_path:", style_path)
    print("out_path:", out_path)

    content = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor()
        ])(Image.open(content_path))
    
    style = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor()
        ])(Image.open(style_path))
    
    generated = fastnst(content.unsqueeze(0), style.unsqueeze(0))

    out_image = transforms.ToPILImage()(generated[0])
    out_image.save(out_path)
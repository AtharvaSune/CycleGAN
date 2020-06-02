import os
import torch
import random
from PIL import Image
import numpy as np


def log_images(obj, epoch):
    root = "./images"
    try:
        os.makedirs(root)
    except:
        pass

    epoch = obj["epoch"]
    batch = obj["batch"]
    realA = obj["images"]["realA"]
    realB = obj["images"]["realB"]
    fakeA = obj["images"]["fakeA"]
    fakeB = obj["images"]["fakeB"]

    os.makedirs(os.path.join(root, str(epoch), "real"), exist_ok=True)
    for i, im in enumerate(realA):
        im = torch.unsqueeze(im, 0)
        im = im.cpu().numpy()
        im = Image.fromarray(im)
        im.save(os.path.join(root, str(epoch), "real", str(i+1)))


def save_model(model, epoch):
    root = "./model"
    model_name = str(model) + "_" + str(epoch)
    try:
        os.makedirs(root)
    except:
        pass

    torch.save({
        "model": model,
        "state_dict": model.state_dict(),
    }, os.path.join(root, model_name))


class Buffer(object):
    def __init__(self, max_size, batch_size=32):
        self.max_size = max_size
        self.ret = []
        self.batch_size = batch_size

    def replay_fake(element):
        element = element.data
        for el in element:
            el = torch.unsqueeze(el, 0)
            if len(self.ret) < self.max_size:
                self.ret.append(el)
            else:
                idx = torch.randint(0, self.max_size)
                self.ret[idx] = el

            random.shuffle(self.ret)

        return torch.cat(self.ret[:self.batch_size])

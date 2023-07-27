import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional


class EvalCounter:

    def __init__(self):
        self.counts = []
        self.current_counts = 0

    def add_counts(self, counts=1):
        self.current_counts += counts

    def new_counter(self):
        self.counts.append(self.current_counts)
        self.current_counts = 0

    def get_average(self):
        return np.mean(np.array(self.counts))

# the goal of GFM is to maximize the margin using gradient ascent
class MarginLoss(torch.nn.Module):

    def __init__(self, reduce=False, margin=0, target=False):
        super(MarginLoss, self).__init__()
        self.reduce = reduce
        self.margin = margin
        self.target = target

    def forward(self, logits, label):
        logit_label = torch.empty_like(label, dtype=logits.dtype, device=logits.device)
        logit_flag = torch.ones_like(logits, dtype=torch.uint8, device=logits.device)
        for i in range(len(label)):
            logit_label[i] = logits[i, label[i]]
            logit_flag[i, label[i]] = 0

        diff = logit_label - torch.max(logits[logit_flag].view(len(label), -1), dim=1)[0]
        margin = -torch.nn.functional.relu(diff + self.margin, True) + self.margin
        return margin

def frozen_params(model):
    for params in model.parameters():
        params.requires_grad = False

# projection the image onto constaint set, clip the value into [0,1], normalize with default (mean,std) 
def clip(origin, image, ell, _min, _max, scale, l2=False):
    noise = image - origin
    if l2:
        for i in range(len(origin)):
            noise[i] = noise[i]/scale[i]
        norm = torch.norm(noise)
        if norm>ell:
            noise = noise/norm*ell
        for i in range(len(origin)):
            noise[i] = noise[i]*scale[i]
        image = origin+noise
    else:
        for i in range(len(origin)):
            noise[i] = torch.min(noise[i], torch.tensor(ell*scale[i], device=noise.device).expand_as(noise[i]))
            noise[i] = torch.max(noise[i], -torch.tensor(ell*scale[i], device=noise.device).expand_as(noise[i]))
        image = origin + noise
    for i in range(len(origin)):
        image[i] = torch.clamp(image[i], _min[i], _max[i])

    return image

# calculate loss for for samples: images, suggest setting args.batch > len(images) to put all samples in a same batch
def calculate_function(model, images, label, criterion, counter, args):
    device = images.device

    n = len(images)
    k = 0
    loss = torch.zeros(n, dtype=torch.float32).to(device)

    while k < n:
        start = k
        end = min(k + args.batch_size, n)
        logits = model(images[start:end])
        loss[start:end] = criterion(logits, label.expand(end - start))
        k = end

    counter.add_counts(len(images))

    return loss

def GFM(model, image, label, criterion, scale, mean, counter, args):
    device = image.device

    lr = args.lr   
    iters = args.max_query // 2

    size = image.size()
    dimension = size[0] * size[1] * size[2]
    noise = torch.empty((dimension, 2), device=device)
    origin_image = torch.tensor(image, device=device)
    _min = -mean*scale
    _max = (1-mean)*scale
    last_F = []
     
    for n in range(iters):
        logit = model(image.view(1, size[0], size[1], size[2]))
        success = torch.argmax(logit, dim=1) !=label
        loss = criterion(logit, label)
        last_F.append(loss)
        last_F = last_F[-args.plateau_length:]

        # using weight decay if no improvement on the loss
        if last_F[-1] < last_F[0]  and len(last_F) == args.plateau_length:
            if lr > args.lr_min:
                print("[log] Annealing max_lr")
                lr = max(lr / args.lr_decay, args.lr_min)
            last_F = []
            
        if bool(success[0]):
            break
        if counter.current_counts > args.max_query:
            break
        
        # generate random vector on uniform sphere
        nn.init.normal_(noise)
        noise = torch.nn.functional.normalize(noise,p=2,dim=0)
        noise[:,1:] = -noise[:, :1]
        images = image.repeat(2, 1, 1, 1) + noise.transpose(0, 1).view(-1, size[0], size[1], size[2]) * args.delta

        diff = calculate_function(model, images, label, criterion, counter, args) / args.delta * dimension
        grad = torch.mean(diff.expand(dimension, 2) * noise, dim=1).view_as(image)
        if n % 20 == 0 and args.show_loss:
            print("iteration: {} loss: {}, success: {}, l2_deviation {}".format(n, float(loss[0]), bool(success[0]), float(torch.norm(image - origin_image))/np.mean(scale)))

        image = clip(origin_image, image + lr * grad, args.ell, _min, _max, scale, args.l2)

    logit = model(image.view(1, size[0], size[1], size[2]))
    success = torch.argmax(logit, dim=1) != label

    return success, image

def GFM_Plus(model, image, label, criterion, scale, mean, counter, args):
    device = image.device

    lr = args.lr   

    size = image.size()
    dimension = size[0] * size[1] * size[2]

    # determine the mega/mini batch size for estimating gradient
    noise_mega = torch.empty((dimension, 2*args.b*args.m), device=device)
    noise_mini = torch.empty((dimension, 2*args.b), device=device)
    origin_image = torch.tensor(image, device=device)
    _min = -mean*scale
    _max = (1-mean)*scale
    last_F = []
    
    n = 0
    while True:
        logit = model(image.view(1, size[0], size[1], size[2]))
        success = torch.argmax(logit, dim=1) !=label
        loss = criterion(logit, label)
        last_F.append(loss)
        last_F = last_F[-args.plateau_length:]

        # using weight decay if no improvement on the loss
        if last_F[-1] < last_F[0]  and len(last_F) == args.plateau_length:
            if lr > args.lr_min:
                print("[log] Annealing max_lr")
                lr = max(lr / args.lr_decay, args.lr_min)
            last_F = []
            
        if bool(success[0]):
            break
        if counter.current_counts > args.max_query:
            break
        
        if n % args.m == 0:
            nn.init.normal_(noise_mega)
            noise = torch.nn.functional.normalize(noise_mega,p=2,dim=0)
        else:
            nn.init.normal_(noise_mini)
            noise = torch.nn.functional.normalize(noise_mini,p=2,dim=0)
            
        # use two point estimate for derivatve at current point
        q = noise.size()[1] // 2
        noise[:,q:] = -noise[:, :q]
        images = image.repeat(2*q, 1, 1, 1) + noise.transpose(0, 1).view(-1, size[0], size[1], size[2]) * args.delta
        diff = calculate_function(model, images, label, criterion, counter, args) / args.delta * dimension
        g = torch.mean(diff.expand(dimension, 2*q) * noise, dim=1).view_as(image)
        
        if n % args.m == 0:
            grad = g
        else:
            # use recursive gradient estimate with previous point
            images = pre_image.repeat(2*q, 1, 1, 1) + noise.transpose(0, 1).view(-1, size[0], size[1], size[2]) * args.delta
            diff = calculate_function(model, images, label, criterion, counter, args) / args.delta * dimension
            pre_g = torch.mean(diff.expand(dimension, 2*q) * noise, dim=1).view_as(image)
            grad = grad + g - pre_g

        if n % 20 == 0 and args.show_loss:
            print("iteration: {} loss: {}, success: {}, l2_deviation {}".format(n, float(loss[0]), bool(success[0]), float(torch.norm(image - origin_image))/np.mean(scale)))
        image = clip(origin_image, image + lr * grad, args.ell, _min, _max, scale, False)
        
        # store the previous image, need not to use copy/deepcopy since it would not be changed
        pre_image = image 
        n = n + 1

    logit = model(image.view(1, size[0], size[1], size[2]))
    success = torch.argmax(logit, dim=1) != label

    return success, image

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
import torch.utils.data
from model.LeNet import LeNet
import os
from utils import *
import numpy as np
import time
import random

parser = argparse.ArgumentParser()
parser.add_argument('--save_prefix', default=None, help='saving path: can using to be opt by default') 
parser.add_argument('--batch_size', type=int, default=512, help='maximum batch size for training')
parser.add_argument('--model_dir', default='./weight', help='path for model')
parser.add_argument('--data_dir', default='/home/truenobility303/truenobility303/Data', help='dataset path for MNIST')
parser.add_argument('--device', default='cuda:0', help='device for training')
parser.add_argument('--ell', type=float, default=0.2, help='l_infinity norm for the perturbation')
parser.add_argument('--max_query', type=int, default=50000, help='iterations for gradient ascent')
parser.add_argument('--delta', type=float, default=0.01, help='difference distance for estimation of gradient')
parser.add_argument('--show_loss', type=int, default=1, help='showing loss when attacking')
parser.add_argument('--b', type=int, default=50, help = 'number of samples for estimate gradient')
parser.add_argument('--m', type=int, default=10, help = 'epoch length')
# learning rates setups
parser.add_argument('--lr', type=float, default=0.5, help='learning rate for gradient ascent')
parser.add_argument('--lr_min', type=float, default=5e-3)
parser.add_argument('--lr_decay', type=float, default=2, help='weight decay: 1 for not used') 
parser.add_argument('--plateau_length', type=int, default=10)
parser.add_argument('--momentum', type=float, default=0.0, help='momentum: 0 for not used') 
# experiment settings
parser.add_argument('--opt', type=str, default='GFM+', help='optimizer') 
parser.add_argument('--seed', type=int, default=42, help='random seed') 
args = parser.parse_args()

# print the arguments
for arg in vars(args):
    print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))

def initialize(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

initialize(args.seed)
if args.opt == 'GFM':
    args.b = args.b * args.m
    args.m = 1

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

mean = np.array([0.1307])
std = np.array([0.3081])

scale = np.array([1 / 0.3081])

test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)])

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root=args.data_dir, train=False, transform=test_transform, download=False),
    batch_size=1, shuffle=False,
    num_workers=1, pin_memory=True)

model = LeNet()
criterion = MarginLoss(margin=4.0)

loaded_state_dict = torch.load(os.path.join(args.model_dir, 'LeNet.pth'))
model.load_state_dict(loaded_state_dict)
model.to(device)

frozen_params(model)

model.eval()
count_success = 0
count_total = 0

if not os.path.exists('log'):
    os.mkdir('log')

counter = EvalCounter()
begin_time = time.time()

# N = len(test_loader)
# B = N // 5
# begin_index = args.n * B
# end_index =  (args.n+1) * B

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        #if i >= begin_index and i < end_index:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)

        correct = torch.argmax(logits, dim=1) == labels

        if correct:
            counter.new_counter()
            count_total += float(correct)

            success, adv = GFM_Plus(model, images[0], labels, criterion, scale, mean, counter, args)
            print("image: {} eval_count: {} success: {}".format(i, counter.current_counts, bool(success[0])))
            count_success += float(success)
            
        if (i+1) % 100 == 0:
            success_rate = float(count_success) / float(count_total)
            avg_time = (time.time() - begin_time) / i
            print("average running time {}".format(avg_time))
            print("success rate {}".format(success_rate))
            print("average eval count {}".format(counter.get_average()))

success_rate = float(count_success) / float(count_total)
np.save(os.path.join('log', '{}_{}_{}_{:5f}.npy'.format(args.seed, args.m, args.b, success_rate)),
            np.array(counter.counts))

# This material was prepared as an account of work sponsored by an agency of the
# United States Government.  Neither the United States Government nor the United
# States Department of Energy, nor Battelle, nor any of their employees, nor any
# jurisdiction or organization that has cooperated in the development of these
# materials, makes any warranty, express or implied, or assumes any legal
# liability or responsibility for the accuracy, completeness, or usefulness or
# any information, apparatus, product, software, or process disclosed, or
# represents that its use would not infringe privately owned rights. Reference
# herein to any specific commercial product, process, or service by trade name,
# trademark, manufacturer, or otherwise does not necessarily constitute or imply
# its endorsement, recommendation, or favoring by the United States Government
# or any agency thereof, or Battelle Memorial Institute. The views and opinions
# of authors expressed herein do not necessarily state or reflect those of the
# United States Government or any agency thereof.
#                 PACIFIC NORTHWEST NATIONAL LABORATORY
#                            operated by
#                             BATTELLE
#                             for the
#                   UNITED STATES DEPARTMENT OF ENERGY
#                    under Contract DE-AC05-76RL01830

import os
import torch
from torch.utils.data.dataloader import DataLoader
from utils import SDataset
from utils import MSE, EdgeWiseKL, MissRate
import csv
import numpy as np
import argparse

parser = argparse.ArgumentParser()
## directories
parser.add_argument('--data', type=str, default='./processed/test.npy', help='processed data file for test set')
parser.add_argument('--save_dir', type=str, default='./results/', help='path to save')
## dataset params
parser.add_argument('--n_atoms', type=int, default=19, help='number of atoms in system')
parser.add_argument('--window_size', type=int, default=10, help='window size')
## model params
parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
parser.add_argument('--model', type=str, default='./results/generator.pkl', help='path to pretrained model')
parser.add_argument('--gpu', type=int, default=0, help='Start Batch')

args = parser.parse_args()

if args.gpu != -1:
    device_str = 'cuda:' + str(args.gpu)
else:
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    device_str = 'cpu'

device = torch.device(device_str)


# Make the save directory
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

save_path = args.save_dir
# Create data loader
test_data = SDataset(args.data)
print(f'data loaded from {args.data}')
print(f'total samples in training set: {test_data.__len__()}')

if args.gpu != -1: torch.cuda.cudart().cudaProfilerStart()

if args.gpu != -1: torch.cuda.nvtx.range_push("DataLoader")
test_loader = DataLoader(
    dataset=test_data,
    batch_size=args.batch_size,
    shuffle=False,
    pin_memory=True
)
if args.gpu != -1: torch.cuda.nvtx.range_pop()


if args.gpu != -1: torch.cuda.nvtx.range_push("load pretrained model")
# load pretrained model
generator = torch.load(args.model, map_location=device)
generator = generator
print(f'model loaded from {args.model}')
if args.gpu != -1: torch.cuda.nvtx.range_pop()

total_samples = 0
total_mse = 0
total_kl = 0
total_missrate = 0



for i, data in enumerate(test_loader):
    if args.gpu != -1: torch.cuda.nvtx.range_push("Forward - " + str(i) ) ## -----> PROFILING

    in_shots, out_shot = data

    if args.gpu != -1: torch.cuda.nvtx.range_push("Send (in_shots, out_shot) toDevice" ) ## -----> PROFILING
    in_shots, out_shot = in_shots.to(device), out_shot.to(device)
    if args.gpu != -1: torch.cuda.nvtx.range_pop() ## -----> PROFILING
    
    if args.gpu != -1: torch.cuda.nvtx.range_push("generator(in_shots)" )
    predicted_shot = generator(in_shots)
    if args.gpu != -1: torch.cuda.nvtx.range_pop()

    if args.gpu != -1: torch.cuda.nvtx.range_push("predicted_shot")
    predicted_shot = predicted_shot.view(-1, args.n_atoms, args.n_atoms)
    if args.gpu != -1: torch.cuda.nvtx.range_pop()

    # average to make symmetric Aij = Aji
    predicted_shot = (predicted_shot + predicted_shot.transpose(1, 2)) / 2
    # put 0 on the diagnal (no self-loops)
    for j in range(args.n_atoms):
        predicted_shot[:, j, j] = 0
    batch_size = in_shots.size(0)
    total_samples += batch_size
    total_mse += batch_size * MSE(predicted_shot, out_shot)
    total_kl += batch_size * EdgeWiseKL(predicted_shot, out_shot)
    total_missrate += batch_size * MissRate(predicted_shot, out_shot)

    if args.gpu != -1: torch.cuda.nvtx.range_push("predicted_shot.cpu()")
    predicted_shot = predicted_shot.cpu().detach().numpy()
    out_shot = out_shot.cpu().detach().numpy()
    if i == 0:
        all_predicted_shots = predicted_shot
        all_true_shots = out_shot
    else:
        all_predicted_shots = np.vstack((all_predicted_shots, predicted_shot))
        all_true_shots = np.vstack((all_true_shots, out_shot))

    if args.gpu != -1: torch.cuda.nvtx.range_pop()

    if args.gpu != -1: torch.cuda.nvtx.range_pop() ## -----> PROFILING

if args.gpu != -1: torch.cuda.cudart().cudaProfilerStop()

# save true and predicted normalized adjacency matrices
np.save(os.path.join(save_path, 'true.npy'), all_true_shots)
np.save(os.path.join(save_path, 'pred.npy'), all_predicted_shots)


# print and save statistics
print('MSE: %.4f' % (total_mse / total_samples))
print('edge wise KL: %.4f' % (total_kl / total_samples))
print('miss rate: %.4f' % (total_missrate / total_samples))

with open(os.path.join(save_path, f'test_set_statistics.csv'), mode='a') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Dataset','Model','MSE','Edge-wise KL','Miss rate'])
    writer.writerow([args.data, args.model,(total_mse / total_samples),(total_kl / total_samples),(total_missrate / total_samples)])
import numpy as np
import os, argparse, time
from tqdm import tqdm

import matplotlib; matplotlib.use('Agg')

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from correspondence import config, data
from correspondence.checkpoints import CheckpointIO
from correspondence.utils import visualize as vis
from correspondence.utils.common import NAMES2ID
from correspondence.utils.auxillary import plant_seeds

torch.multiprocessing.set_sharing_strategy('file_system')
plant_seeds(random_seed=8888)


parser = argparse.ArgumentParser(
    description='Evaluate CPAE.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('-j', type=int, default=16)
parser.add_argument('-c',
                    choices=['airplane', 'bag', 'bench', 'bathtub', 'bed', 'bottle', 'cap', 'car', 'chair', 'earphone', 'guitar',
                             'helmet', 'knife', 'lamp', 'laptop', 'motorcycle', 'mug', 'pistol', 'rifle', 'rocket', 'couch',
                             'skateboard', 'table', 'vessel'], # not all yet
                    help='Specify Category')
parser.add_argument('--load', type=str, help='Path to weight file.')
parser.add_argument('--tag', type=str, default='', help='special running tag')

# Get configuration and basic arguments
args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Shorthands
category_name = args.c
category_id = NAMES2ID[args.c]
base_dir = cfg['training']['out_dir']

# Output directory
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
if len(args.tag) > 0:
    args.tag = '_' + args.tag
out_dir = os.path.join(base_dir, category_id+'_'+category_name+args.tag)

# Dataset

dataset = config.get_dataset('test', cfg, categories=[category_id])
model = config.get_model(cfg, device=device, dataset=dataset)

checkpoint_io = CheckpointIO(out_dir, model=model)
try:
    # checkpoint_io.load(cfg['test']['model_file'])
    load_dict = checkpoint_io.load(args.load, strict=True)
except FileExistsError:
    print('Model file does not exist. Exiting.')
    exit()

# Trainer
trainer = config.get_trainer(model, None, out_dir, cfg, device=device)

# Evaluate
model.eval()

eval_dicts = []   
print('Evaluating networks...')

test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=32, num_workers=args.j, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

# Handle each dataset separately
eval_dict = trainer.evaluate(test_loader, epoch_it=0, save=out_dir)
import numpy as np
import os, argparse, time
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

# Arguments
parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--tag', type=str, default='', help='special running tag')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
# parser.add_argument('--no-vis', action='store_true', help='Do not visualize.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')
parser.add_argument('-j', type=int, default=16)
parser.add_argument('-c',
                    choices=['airplane', 'bag', 'bench', 'bathtub', 'bed', 'bottle', 'cap', 'car', 'chair', 'earphone', 'guitar',
                             'helmet', 'knife', 'lamp', 'laptop', 'motorcycle', 'mug', 'pistol', 'rifle', 'rocket', 'couch',
                             'skateboard', 'table', 'vessel'], # not all yet
                    help='Specify Category')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")

# Set t0
t0 = time.time()

# Shorthands
category_name = args.c
category_id = NAMES2ID[args.c]
base_dir = cfg['training']['out_dir']
batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
exit_after = args.exit_after

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

if len(args.tag) > 0:
    args.tag = '_' + args.tag

out_dir = os.path.join(base_dir, category_id+'_'+category_name+args.tag)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Dataset
train_dataset = config.get_dataset('train', cfg, categories=[category_id])
val_dataset = config.get_dataset('val', cfg, categories=[category_id])
test_dataset = config.get_dataset('test', cfg, categories=[category_id])

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=args.j, shuffle=True, drop_last=True,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, num_workers=args.j, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, num_workers=args.j, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)

# For visualizations
vis_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, num_workers=args.j, shuffle=False,
    collate_fn=data.collate_remove_none,
    worker_init_fn=data.worker_init_fn)
data_vis = next(iter(vis_loader))

# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)

# Intialize training
optimizer = optim.Adam(model.parameters(), lr=1e-4)
trainer = config.get_trainer(model, optimizer, out_dir, cfg, device=device)
checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)

# Load weight
try:
    if cfg['method'] == 'unfoldnet':
        load_dict = checkpoint_io.load('model_best.pt', strict=True)
    else:
        load_dict = checkpoint_io.load('model.pt', strict=True)
except FileExistsError:
    load_dict = dict()

# Load metric
epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', -1)
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))

# Initial Logger
logger = SummaryWriter(os.path.join(out_dir, 'logs'))

# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']
min_val_epoch = cfg['training']['min_val_epoch']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print(model)
print('Total number of parameters: %d' % nparameters)

smoothed_total_loss = 0

while True:
    epoch_it += 1

    for batch in train_loader:
        it += 1
        loss = trainer.train_step(batch, epoch_it=epoch_it, it=it)
        scalars = trainer.get_current_scalars()
        vis.tsboard_log_scalar(logger, scalars, it)
        logger.add_scalar('train/loss', loss, it)
        # Print output
        if print_every > 0 and (it % print_every) == 0:
            vis.print_current_scalars(epoch_it, it, scalars)

        # Visualize output
        if visualize_every > 0 and (it % visualize_every) == 0 and it != 0 and not args.no_cuda:
        # if True:
            print('Visualizing')
            trainer.visualize(data_vis, logger=logger, it=it, epoch_it=epoch_it)

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            print('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
        # Run validation
        if validate_every > 0 and (it % validate_every) == 0 and epoch_it >= min_val_epoch:
            eval_dict = trainer.evaluate(val_loader, epoch_it=epoch_it)
            metric_val = eval_dict[model_selection_metric]
            print('Validation metric (%s): %.4f'
                  % (model_selection_metric, metric_val))

            for k, v in eval_dict.items():
                logger.add_scalar('val/%s' % k, v, it)

            for k, v in eval_dict.items():
                print('%s : %.5f' % (k, v))

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                print('New best model (loss %.4f)' % metric_val_best)
                checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)
                
                # evaluate on test set
                eval_dict = trainer.evaluate(test_loader, epoch_it=epoch_it, save=out_dir)

        # Exit if necessary
        if exit_after > 0 and (time.time() - t0) >= exit_after:
            print('Time limit reached. Exiting.')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            exit(3)

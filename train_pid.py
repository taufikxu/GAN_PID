import utils_log
from gan_training.config import (
    load_config,
    build_models,
    build_optimizers,
    build_lr_scheduler,
)
from gan_training.eval import Evaluator
from gan_training.distributions import get_ydist, get_zdist
from gan_training.inputs import get_dataset
from gan_training.checkpoints import CheckpointIO
from gan_training.logger import Logger
from gan_training.train_pid import Trainer, update_average
from gan_training import utils
from torch import nn
import shutil
import copy
import time
from os import path
import os
import argparse
import torch
import numpy as np

torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# Arguments
parser = argparse.ArgumentParser(
    description='Train a GAN with different regularization strategies.')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--oldmodel', type=str, help='Path to previous file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('-key', type=str, default='', help='')
args = parser.parse_args()

config = load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
config['training']['nworkers'] = 16
config['training']['backup_every'] = 3600 * 8
# config['training']['iv'] = 0

# Short hands
batch_size = config['training']['batch_size']
d_steps = config['training']['d_steps']
restart_every = config['training']['restart_every']
inception_every = config['training']['inception_every']
save_every = config['training']['save_every']
backup_every = config['training']['backup_every']
sample_nlabels = config['training']['sample_nlabels']

out_dir = "{}{}_{}_{}".format(config['training']['out_dir'],
                              time.strftime("%Y-%m-%d-%H-%M-%S"),
                              config['training']['out_basename'], args.key)
checkpoint_dir = path.join(out_dir, 'chkpts')

# Create missing directories
if not path.exists(out_dir):
    os.makedirs(out_dir)
if not path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
shutil.copy(args.config, os.path.join(out_dir, "config.yaml"))

# Logger
checkpoint_io = CheckpointIO(checkpoint_dir=checkpoint_dir)

device = torch.device("cuda:0" if is_cuda else "cpu")

# Dataset
train_dataset, nlabels = get_dataset(
    name=config['data']['type'],
    data_dir=config['data']['train_dir'],
    size=config['data']['img_size'],
    lsun_categories=config['data']['lsun_categories_train'],
    config=config)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=config['training']['nworkers'],
    shuffle=True,
    pin_memory=True,
    sampler=None,
    drop_last=True)
# toy_data = config['data']['type'].lower() in ['mog']

# Number of labels
nlabels = min(nlabels, config['data']['nlabels'])
sample_nlabels = min(nlabels, sample_nlabels)

# Create models
generator, discriminator = build_models(config)
print(generator)
print(discriminator)

# Put models on gpu if needed
generator = generator.to(device)
discriminator = discriminator.to(device)

g_optimizer, d_optimizer = build_optimizers(generator, discriminator, config)

# Use multiple GPUs if possible
generator = nn.DataParallel(generator)
discriminator = nn.DataParallel(discriminator)

# Register modules to checkpoint
checkpoint_io.register_modules(
    generator=generator,
    discriminator=discriminator,
    g_optimizer=g_optimizer,
    d_optimizer=d_optimizer,
)

# Get model file
model_file = config['training']['model_file']

# Logger
logger = Logger(log_dir=path.join(out_dir, 'logs'),
                img_dir=path.join(out_dir, 'imgs'),
                monitoring=config['training']['monitoring'],
                monitoring_dir=path.join(out_dir, 'monitoring'))

text_logger = utils_log.build_logger(out_dir)

# Distributions
ydist = get_ydist(nlabels, device=device)
zdist = get_zdist(config['z_dist']['type'],
                  config['z_dist']['dim'],
                  device=device)

ntest = batch_size
x_real_test, ytest = utils.get_nsamples(train_loader, ntest)
ytest.clamp_(None, nlabels - 1)
ztest = zdist.sample((ntest, ))
utils.save_images(x_real_test, path.join(out_dir, 'real.png'))

# Test generator
if config['training']['take_model_average']:
    generator_test = copy.deepcopy(generator)
    checkpoint_io.register_modules(generator_test=generator_test)
else:
    generator_test = generator

# Evaluator
evaluator = Evaluator(generator_test,
                      zdist,
                      ydist,
                      batch_size=batch_size,
                      device=device)

# Train
tstart = t0 = time.time()

# Load checkpoint if it exists
try:
    load_dict = checkpoint_io.load(model_file)
except FileNotFoundError:
    it = epoch_idx = -1
    print("No loaded model, from initialization")
    evaluation_flag = False
else:
    print("successfully loaded")
    evaluation_flag = False
    it = load_dict.get('it', -1)
    epoch_idx = load_dict.get('epoch_idx', -1)
    logger.load_stats('stats.p')

# Reinitialize model average if needed
if (config['training']['take_model_average']
        and config['training']['model_average_reinit']):
    update_average(generator_test, generator, 0.)

# Learning rate anneling
g_scheduler = build_lr_scheduler(g_optimizer, config, last_epoch=it)
d_scheduler = build_lr_scheduler(d_optimizer, config, last_epoch=it)

# Trainer
trainer = Trainer(generator,
                  discriminator,
                  g_optimizer,
                  d_optimizer,
                  gan_type=config['training']['gan_type'],
                  reg_type=config['training']['reg_type'],
                  reg_param=config['training']['reg_param'],
                  pv=config['training']['pv'],
                  iv=config['training']['iv'],
                  dv=config['training']['dv'],
                  batch_size=config['training']['batch_size'],
                  config=config)

# Training loop
print('Start training...')
while epoch_idx < 1600:
    epoch_idx += 1
    print('Start epoch %d...' % epoch_idx)

    for x_real, y in train_loader:
        it += 1
        g_scheduler.step()
        d_scheduler.step()

        d_lr = d_optimizer.param_groups[0]['lr']
        g_lr = g_optimizer.param_groups[0]['lr']
        logger.add('learning_rates', 'discriminator', d_lr, it=it)
        logger.add('learning_rates', 'generator', g_lr, it=it)

        x_real, y = x_real.to(device), y.to(device)
        y.clamp_(None, nlabels - 1)

        # Discriminator updates
        z = zdist.sample((batch_size, ))
        dloss, dl, il = trainer.discriminator_trainstep(x_real, y, z, it)
        logger.add('losses', 'discriminator', dloss, it=it)
        logger.add('losses', 'd_loss', dl, it=it)
        logger.add('losses', 'i_loss', il, it=it)

        # Generators updates
        if ((it + 1) % d_steps) == 0:
            z = zdist.sample((batch_size, ))
            gloss = trainer.generator_trainstep(y, z)
            logger.add('losses', 'generator', gloss, it=it)

            if config['training']['take_model_average']:
                update_average(generator_test,
                               generator,
                               beta=config['training']['model_average_beta'])

        # Print stats
        if it % 100 == 0:
            g_loss_last = logger.get_last('losses', 'generator')
            d_loss_last = logger.get_last('losses', 'discriminator')
            dl_last = logger.get_last('losses', 'd_loss')
            il_last = logger.get_last('losses', 'i_loss')
            text_logger.info(
                '[epoch %0d, it %4d] g_loss = %9.4f, d_loss = %9.4f, dl=%9.4f, il=%9.4f'
                % (epoch_idx, it, g_loss_last, d_loss_last, dl_last, il_last))

        # (i) Sample if necessary
        if (it % config['training']['sample_every']) == 0:
            print('Creating samples...')
            x = evaluator.create_samples(ztest, ytest)
            logger.add_imgs(x, 'all', it)
            for y_inst in range(sample_nlabels):
                x = evaluator.create_samples(ztest, y_inst)
                logger.add_imgs(x, '%04d' % y_inst, it)

        # (ii) Compute inception if necessary
        if evaluation_flag is True or (inception_every > 0 and
                                       ((it + 1) % inception_every) == 0):
            evaluation_flag = False
            inception_mean, inception_std = evaluator.compute_inception_score()
            logger.add('inception_score', 'mean', inception_mean, it=it)
            logger.add('inception_score', 'stddev', inception_std, it=it)
            text_logger.info(
                '[epoch %0d, it %4d] inception_mean: %.4f, inception_std: %.4f'
                % (epoch_idx, it, inception_mean, inception_std))

        # (iii) Backup if necessary
        if ((it + 1) % backup_every) == 0:
            text_logger.info('Saving backup...')
            checkpoint_io.save('model_%08d.pt' % it, it=it)
            logger.save_stats('stats_%08d.p' % it)

        # (iv) Save checkpoint if necessary
        # if time.time() - t0 > save_every:
        #     text_logger.info('Saving checkpoint...')
        #     checkpoint_io.save(model_file, it=it)
        #     logger.save_stats('stats.p')
        #     t0 = time.time()

        #     if (restart_every > 0 and t0 - tstart > restart_every):
        #         exit(3)

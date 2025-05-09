import click
from os.path import join, dirname, abspath
import subprocess
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import yaml

import tarl.datasets.datasets as datasets
import tarl.models.models as models
import os
from pathlib import Path

@click.command()
### Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'config/config.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt). Use this flag if you just want to load the weights from the checkpoint file without resuming training.',
              default=None)
@click.option('--checkpoint',
              '-ckpt',
              type=str,
              help='path to checkpoint file (.ckpt) to resume training.',
              default=None)
def main(config,weights,checkpoint):
    cfg = yaml.safe_load(open(config))
    cfg['git_commit_version'] = str(subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']).strip())

    #Load data and model
    data = datasets.data_modules[cfg['data']['dataloader']](cfg)
    trainer_type = cfg['experiment'].get('trainer', 'TARLTrainer')
    # model = models.StatNet(cfg)
    if weights is None:
        if trainer_type == 'TARLTrainer':
            model = models.TARLTrainer(cfg, data)
        elif trainer_type == 'OneWayTARL':
            model = models.OneWayTARLTrainer(cfg, data)
    else:
        print('Loading: ', weights)
        ckpt = torch.load(weights)
        trainer_type = ckpt['hyper_parameters']['experiment']['trainer']
        if trainer_type == 'TARLTrainer':
            model = models.TARLTrainer.load_from_checkpoint(weights,hparams=cfg)
        elif trainer_type == 'OneWayTARL':
            model = models.OneWayTARLTrainer.load_from_checkpoint(weights,hparams=cfg)
        model_save_path = os.path.splitext(Path(weights).name)[0]
        model.save_backbone(model_save_path)
        exit()

    #Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_saver = ModelCheckpoint(every_n_epochs=10,
                                 filename=cfg['experiment']['id']+'_{epoch:02d}_{loss:.2f}',
                                 save_top_k=-1,
                                 save_last=True)

    tb_logger = pl_loggers.TensorBoardLogger('experiments/'+cfg['experiment']['id'],
                                             default_hp_metric=False)

    #Setup trainer
    trainer = Trainer(gpus=cfg['train']['n_gpus'],
                      logger=tb_logger,
                      log_every_n_steps=100,
                      resume_from_checkpoint=checkpoint,
                      max_epochs= cfg['train']['max_epoch'],
                      callbacks=[lr_monitor, checkpoint_saver],
                      )#track_grad_norm=True)

    # Train!
    trainer.fit(model, data)

if __name__ == "__main__":
    main()

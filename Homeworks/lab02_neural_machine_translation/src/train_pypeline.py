import sys
import logging

import pynndescent.distances

from data.make_dataset import iterators_and_fields
from entity.train_pipeline_params import read_training_pipeline_params
from models.base_seq2seq import base_seq2seq
import click
from train_model import train_model
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(config_path: str, report_file: str):
    """train pipeline"""
    all_params = read_training_pipeline_params(config_path)

    data_params_dict = vars(all_params.dataparams)
    data, fields = iterators_and_fields(**data_params_dict)
    train_iterator, valid_iterator, test_iterator = data
    SRC, TRG = fields
    base_model_params = vars(all_params.seq2seqparams)
    logger.info("Try build the baseline model...")
    base_model = base_seq2seq(len(SRC.vocab), len(TRG.vocab), **base_model_params)
    logger.info("baseline loaded!!!")
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(base_model.parameters())
    train_params = all_params.trainparams
    train_model(base_model,
                optimizer,
                criterion,
                train_iterator,
                valid_iterator,
                report_file,
                train_params.clip,
                train_params.num_epochs,
                )

@click.command(name='train_pipeline')
@click.argument('config_path', default='configs/train_config.yml')
@ click.argument('report_file', default='reports/baseline')
def train_pipeline_command(config_path: str):
    """ Make start for terminal """
    train_pipeline(config_path)


if __name__ == '__main__':
    train_pipeline_command()
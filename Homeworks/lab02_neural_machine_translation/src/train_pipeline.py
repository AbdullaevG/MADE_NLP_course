import sys
import logging

import pynndescent.distances

from data.make_dataset import iterators_and_fields
from entity.train_pipeline_params import read_params
from models.base_seq2seq import base_seq2seq
from models.seq2seq_attention import seq2seq_attention
from models.conv_seq2seq import conv_seq2seq
import click
from train_model import train_model
import torch
import torch.nn as nn
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model(model_type: str):
    if model_type == "baseline":
        return base_seq2seq
    elif model_type == "seq2seq_attention":
        return seq2seq_attention
    elif model_type == "conv_seq2seq":
        return conv_seq2seq
    else:
        logger.exception('Model name is incorrect')
        raise NotImplementedError()


def train_pipeline(model_name: str, data_config_path: str, model_config_path: str, report_file: str, translated_exams_file: str):
    """train pipeline"""
    dataparams = read_params(data_config_path, key="data_params")
    data_params_dict = vars(dataparams.dataparams)
    print(data_params_dict)
    data, fields = iterators_and_fields(**data_params_dict)
    train_iterator, valid_iterator, test_iterator = data
    SRC, TRG = fields

    model_params = read_params(model_config_path, key=model_name)
    print("model_params: ", model_params)
    model_params_dict = vars(model_params.modelparams)
    print("model_params_dict: ", model_params_dict)
    logger.info("Try build model...")
    model = get_model(model_params_dict["model_type"])
    model = model(len(SRC.vocab), len(TRG.vocab), **model_params_dict)
    logger.info("model was loaded!!!")


@click.command(name='train_pipeline')
@click.argument('model_name', default='conv_seq2seq')
@click.argument('data_config_path', default='configs/data_config.yml')
@click.argument('model_config_path', default='configs/train_conv_seq2seq.yml')
@click.argument('report_file', default='reports/conv_seq2seq.log')
@click.argument('translated_exams_file', default='reports/conv_seq2seq_generated.txt')
def train_pipeline_command(model_name: str, data_config_path: str, model_config_path: str, report_file: str, translated_exams_file: str):
    """ Make start for terminal """
    train_pipeline(model_name, data_config_path, model_config_path, report_file, translated_exams_file)


if __name__ == '__main__':
    train_pipeline_command()
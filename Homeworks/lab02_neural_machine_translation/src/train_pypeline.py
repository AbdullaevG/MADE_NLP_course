from data.make_dataset import iterators_and_fields
from entity.train_pipeline_params import read_training_pipeline_params
import click


def train_pipeline(config_path: str):
    """train pipeline"""
    all_params = read_training_pipeline_params(config_path)
    data_params_dict = vars(all_params.dataparams)
    data, fields = iterators_and_fields(**data_params_dict)


@click.command(name='train_pipeline')
@click.argument('config_path', default='configs/train_config.yml')
def train_pipeline_command(config_path: str):
    """ Make start for terminal """
    train_pipeline(config_path)


if __name__ == '__main__':
    train_pipeline_command()
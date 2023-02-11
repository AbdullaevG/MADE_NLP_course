"""Training model pipeline"""

from dataclasses import dataclass
from marshmallow_dataclass import class_schema
import yaml
from .data_params import DataParams

@dataclass
class TrainingPipelineParams:
    """Structure for pipeline parameters"""
    dataparams: DataParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str):
    """Read config for model training"""
    with open(path, 'r', encoding='utf-8') as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))


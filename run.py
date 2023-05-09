from lightning.pytorch import Trainer
from lightning.pytorch.utilities.model_summary import ModelSummary
from omegaconf import OmegaConf

from data.data_module import AugmentedDataModule
from utils.parser import parse_arguments
from src.model import Whisper

def main(args):
    config = OmegaConf.load(args.config)

    model = Whisper(config=config)

    trainer = Trainer()
    df = AugmentedDataModule(config.data.root, config=config)
    trainer.fit(model, datamodule=df)

    print('Done')

if __name__ == '__main__':
    args = parse_arguments()

    main(args)
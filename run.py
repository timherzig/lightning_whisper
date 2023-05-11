from omegaconf import OmegaConf
from lightning.pytorch import Trainer

from src.model import Whisper
from utils.parser import parse_arguments
from data.data_module import AugmentedDataModule

def main(args):
    config = OmegaConf.load(args.config)

    model = Whisper(config=config)

    trainer = Trainer(max_epochs=config.train.epochs)
    df = AugmentedDataModule(config.data.root, config=config, batch_size=config.train.batch_size)
    df.setup()

    trainer.fit(model, train_dataloaders=df.train_dataloader())
    
    trainer.test(model, dataloaders=df.test_dataloader())

    print('Done')

if __name__ == '__main__':
    args = parse_arguments()

    main(args)
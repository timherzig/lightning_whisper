import os
from omegaconf import OmegaConf
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger

from src.model import Whisper
from utils.parser import parse_arguments
from data.data_module import AugmentedDataModule


def main(args):
    config = OmegaConf.load(args.config)
    nrun = len(os.listdir("checkpoints/"))
    logger = WandbLogger(
        project="babycry_whisper",
        log_model=True,
        save_dir=f"checkpoints/run_{nrun}",
        name=f"run_{nrun}",
    )  # , config=config)

    model = Whisper(config=config)

    trainer = Trainer(
        max_epochs=config.train.epochs,
        logger=logger,
        accelerator=config.train.accelerator,
        devices=args.gpus,
        num_nodes=args.nodes,
    )
    df = AugmentedDataModule(
        config.data.root,
        config=config,
        batch_size=config.train.batch_size,
        logger=logger,
    )
    df.setup()

    trainer.fit(model, datamodule=df)
    trainer.test(model, datamodule=df)

    print("Done")


if __name__ == "__main__":
    args = parse_arguments()

    main(args)

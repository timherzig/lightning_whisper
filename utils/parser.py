from argparse import ArgumentParser


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        help="location of config file",
        default="config/default_tiny_whisper.yaml",
    )
    parser.add_argument("--gpus", type=int, help="number of gpus to use")
    parser.add_argument("--nodes", type=int, help="number of nodes to use")

    return parser.parse_args()

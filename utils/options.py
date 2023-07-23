import argparse


class Options:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data-root", type=str, default="/home/dell/data/public/SECOND2")
        parser.add_argument("--batch-size", type=int, default=8)
        parser.add_argument("--val-batch-size", type=int, default=16)
        parser.add_argument("--test-batch-size", type=int, default=16)
        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument("--lr", type=float, default=0.0002)
        parser.add_argument('--lr_policy', default='linear', type=str,
                            help='linear | step')
        parser.add_argument("--weight-decay", type=float, default=1e-4)
        parser.add_argument("--model", type=str, default="SCDNet")
        parser.add_argument("--save-mask", dest="save_mask",default="False", action="store_true",
                           help='save predictions of validation set during training')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        print(args)
        return args

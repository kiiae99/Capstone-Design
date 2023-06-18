import argparse
from trainer import Trainer

use_CUDA = True # True for GPU, False for CPU
batch_size = 64 # Batch size
epochs = 60     # Number of epochs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MMAE')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--is_MT', action='store_true', default=False)
    parser.add_argument('--language', type=str, default='koBert')
    parser.add_argument('--use_CUDA', type=bool, default=use_CUDA)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.run()
    pass

if __name__ == "__main__":
    main()
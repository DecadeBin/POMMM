import argparse
import datetime
import os
# model vit #########
# from solver import Solver
# from model_optical_matrix_wo_norm import batch
#####################

# model 2fc #########
from solver_2fc import Solver
from model_2fc import batch
#####################
def main(args):
    os.makedirs(args.model_path, exist_ok=True)

    solver = Solver(args)
    solver.train()
    solver.test()
def main_test(args):
    os.makedirs(args.model_path, exist_ok=True)

    solver = Solver(args)
    solver.test()


def print_args(args):
    for k in dict(sorted(vars(args).items())).items():
        print(k)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=batch)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--log_step', type=int, default=50)

    parser.add_argument('--dset', type=str, default='mnist', help=['mnist', 'fmnist']) # Change the dataset here
    parser.add_argument("--img_size", type=int, default=28, help="Img size")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch Size")
    parser.add_argument("--n_channels", type=int, default=1, help="Number of channels")
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--model_path', type=str, default='./model')

    parser.add_argument("--embed_dim", type=int, default=50, help="dimensionality of the latent space")
    parser.add_argument("--n_attention_heads", type=int, default=5, help="number of heads to be used")
    parser.add_argument("--forward_mul", type=int, default=1, help="forward multiplier")
    parser.add_argument("--n_layers", type=int, default=1, help="number of encoder layers")
    parser.add_argument("--load_model", type=bool, default=False, help="Load saved model")

    start_time = datetime.datetime.now()
    print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))

    args = parser.parse_args()
    args.model_path = os.path.join(args.model_path, args.dset)
    print(args)
    # 训练网络：
    main(args)
    # 测试一个网络性能
    # main_test(args)
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
    print("Duration: " + str(duration))
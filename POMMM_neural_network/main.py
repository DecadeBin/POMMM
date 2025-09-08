import argparse
import datetime
import os
# model vit #########
# from solver import Solver
# from model_optical_matrix_wo_norm import batch
#####################

# model 2fc #########a
from solver import Solver
from model_fun import cal_omm_fix_para

#####################
# 01训练 ommm1900量化
parser = argparse.ArgumentParser(description='Transformer')
parser.add_argument('--dset', type=str, default='fmnist', help=['mnist', 'fmnist'])  # Change the dataset here
parser.add_argument('--exp_type', type=str, default='omm')  # mmm_01,omm,standard_mmm,omm_255,three_stage
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--load_model_type', type=str, default='omm')  # mmm_01,omm,stand
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--only_test', type=bool, default=False)
parser.add_argument('--hook', type=bool, default=False)
parser.add_argument('--three_stage', type=bool, default=False)
parser.add_argument("--stage_time", type=int, default=1, help="")

parser.add_argument("--N", type=int, default=1900, help="")


# ard_mmm,mmm0_255,omm_255,>
parser.add_argument('--model_name', type=str, default='2fc')  # 2fc, vit
args = parser.parse_args()
if args.exp_type == 'standard_mmm' or args.exp_type == 'mmm_01' or args.exp_type == 'three_stage':
    qunatum = False
    omm = False
if args.exp_type == 'omm' or args.exp_type == 'omm_mm01' :
    qunatum = False
    omm = True
if args.exp_type == 'omm_255':
    qunatum = True
    omm = True
if args.exp_type == 'mmm255':
    qunatum = True
    omm = False

def main(args):

    # os.makedirs(args.model_path, exist_ok=True)
    # for i in range(40):
    #     args.N = 50 + i * 50
    solver = Solver(args)
    args.fft_sig_50,args.iter_50,args.full_sig_50,args.fft_sig_1,args.iter_1,args.full_sig_1 = \
        cal_omm_fix_para(args.N)
    if args.only_test==True:
        solver.test()
    else:
        solver.train(args)
        solver.test()



parser.add_argument('--qunatum', type=bool, default=qunatum)  # mmm_01,omm,standard_mmm,mmm0-255
parser.add_argument('-omm', type=bool, default=omm)  # 2fc, vit

parser.add_argument('--n_classes', type=int, default=10)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument("--img_size", type=int, default=28, help="Img size")
parser.add_argument("--patch_size", type=int, default=4, help="Patch Size")
parser.add_argument("--n_channels", type=int, default=1, help="Number of channels")
parser.add_argument('--data_path', type=str, default='./data/')
parser.add_argument('--model_path', type=str, default='./model')
parser.add_argument("--embed_dim", type=int, default=50, help="dimensionality of the latent space")
parser.add_argument("--n_attention_heads", type=int, default=5, help="number of heads to be used")
parser.add_argument("--forward_mul", type=int, default=1, help="forward multiplier")
parser.add_argument("--n_layers", type=int, default=1, help="number of encoder layers")

parser.add_argument("--fft_sig_50", type=float, default=0, help="Patch Size")
parser.add_argument("--iter_50", type=float, default=0, help="Patch Size")
parser.add_argument("--full_sig_50", type=float, default=0, help="Patch Size")
parser.add_argument("--fft_sig_1", type=float, default=0, help="Patch Size")
parser.add_argument("--iter_1", type=float, default=0, help="Patch Size")
parser.add_argument("--full_sig_1", type=float, default=0, help="Patch Size")
parser.add_argument("--g", type=float, default=2, help="Patch Size")
parser.add_argument("--M", type=float, default=50, help="Patch Size")



start_time = datetime.datetime.now()
print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))
args = parser.parse_args()

def repeat_all_exp(args):
    # os.makedirs(args.model_path, exist_ok=True)
    exp_type_list = ['standard_mmm' ,'mmm_01','omm','mmm255']
    model_name_list = ['vit' ,'2fc']
    # dataset_name_llsit = ['']
    for i in range(len(exp_type_list)):
        for j in range(len(model_name_list)):
            args.exp_type = exp_type_list[i]
            args.model_name = model_name_list[j]
            args.load_model_type = exp_type_list[i]
            print('exp_type: ',args.exp_type,'model_name: ',args.model_name)
            if args.exp_type == 'standard_mmm' or args.exp_type == 'mmm_01':
                args.qunatum = False
                args.omm = False
                args.batch_size = 256
                args.epochs = 100
            if args.exp_type == 'omm':
                args.qunatum = False
                args.omm = True
                args.batch_size = 32
                args.epochs = 50
            if args.exp_type == 'mmm255':
                args.qunatum = True
                args.omm = False
                args.batch_size = 256
                args.epochs = 100
            solver = Solver(args)
            solver.train(args)
            solver.test()
def main_test(args):
    # os.makedirs(args.model_path, exist_ok=True)

    solver = Solver(args)
    solver.test()


def print_args(args):
    for k in dict(sorted(vars(args).items())).items():
        print(k)
    print()


if __name__ == '__main__':

    # args.model_path = os.path.join(args.model_path, args.dset)
    print(args)
    # 训练网络：
    # repeat_all_exp(args)
    main(args)
    # 测试一个网络性能
    # main_test(args)
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
    print("Duration: " + str(duration))

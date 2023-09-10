import argparse

def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="train", help="what to do")
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset_name")
    parser.add_argument("--root", type=str, default="./datasets", help="root dataset dir")
    parser.add_argument("--task", type=str, default="train", help="what to do")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_arg()

    !python train.py --dataset cifar10 --use-ema --use-ddim --num-save-images 80 --use-cfg --epochs 600 --chkpt-intv 120 --image-intv 10 --root ./datasets
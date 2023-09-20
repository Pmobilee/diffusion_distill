import os
import json
import torch
from datetime import datetime
from torch.optim import AdamW, lr_scheduler
from v_diffusion import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing import errors
from functools import partial
import wandb
import pandas as pd

def list_files(directory_path):
    with os.scandir(directory_path) as entries:
        return [entry.name for entry in entries if entry.is_file()]
    

def list_folders(directory_path):
    with os.scandir(directory_path) as entries:
        return [entry.name for entry in entries if entry.is_dir()]

@errors.record
def main(args):
    fid_dataframe = pd.DataFrame(columns=['dataset', 'checkpoint', 'fid'])
    chkpts_dir = r"/run/media/damion/Data/Thesis BACKUP/distilled/"
    datasets = list_folders(chkpts_dir)
    
    for dataset in datasets:
        batch_size = 32
        num_workers = 4
        distributed = False
        split = "all" if dataset == "celeba" else "train"
        root =  os.path.expanduser("./datasets")
        trainloader, sampler = get_dataloader(
        dataset=dataset, batch_size=batch_size // args.num_accum, split=split, val_size=0.,
        root=root, drop_last=True, pin_memory=True, num_workers=num_workers, distributed=distributed
    )  # dr
        print(dataset)
        chkpts = list_files(chkpts_dir + dataset + "/")
        # print(chkpts)
        # dataset = args.dataset
        for chkpt in chkpts:
            if not ".pth" in chkpt:
                continue

            try:
                checkpoint_location = chkpts_dir + dataset  + "/" + chkpt
                name = chkpt
                print(checkpoint_location)
                split = "all" if dataset == "celeba" else "train"
                num_workers = args.num_workers
                batch_size = args.batch_size
                in_channels = DATA_INFO[dataset]["channels"]
                image_res = DATA_INFO[dataset]["resolution"]
                image_shape = (in_channels, ) + image_res
                num_classes = DATA_INFO[dataset].get("num_classes", 0)
                w_guide = args.w_guide
                p_uncond = args.p_uncond
                configs_path = os.path.join(args.config_dir, args.dataset + ".json")
                with open(configs_path, "r") as f:
                    configs: dict = json.load(f)
                getdif = partial(get_param, configs_1=configs.get("diffusion", {}), configs_2=args)
                logsnr_schedule = getdif("logsnr_schedule")
                logsnr_min, logsnr_max = getdif("logsnr_min"), getdif("logsnr_max")
                train_timesteps = getdif("train_timesteps")
                sample_timesteps = getdif("sample_timesteps")
                reweight_type = getdif("reweight_type")
                logsnr_fn = get_logsnr_schedule(logsnr_schedule, logsnr_min=logsnr_min, logsnr_max=logsnr_max)
                model_out_type = getdif("model_out_type")
                model_var_type = getdif("model_var_type")
                loss_type = getdif("loss_type")

                diffusion = GaussianDiffusion(
                    logsnr_fn=logsnr_fn,
                    sample_timesteps=sample_timesteps,
                    model_out_type=model_out_type,
                    model_var_type=model_var_type,
                    reweight_type=reweight_type,
                    loss_type=loss_type,
                    intp_frac=args.intp_frac,
                    w_guide=w_guide,
                    p_uncond=p_uncond
                    )

                device = torch.device("cuda:0")

                model = torch.load(checkpoint_location, map_location=device)
                # model = model.to("cuda:0")
                print("loaded")
                optimizer = None
                distill_optimizer = None

                scheduler = None
                chkpt_intv = 1
                image_intv = 1
                num_save_images = 1
        
                train_timesteps = 128
                epochs = None
                grad_norm = None
                image_shape = image_shape
                train_device = None
                rank = None
                distributed = None

            
                trainer = Trainer(
                    model=model,
                    optimizer=optimizer,
                    distill_optimizer=distill_optimizer,
                    diffusion=diffusion,
                    timesteps=train_timesteps,
                    epochs=epochs,
                    trainloader=trainloader,
                    sampler=sampler,
                    shape = image_shape,
                    device = device,
                    scheduler=scheduler,
                    use_cfg=args.use_cfg,
                    use_ema=args.use_ema,
                    grad_norm=grad_norm,
                    num_accum=args.num_accum,
                    chkpt_intv=chkpt_intv,
                    image_intv=image_intv,
                    num_save_images=num_save_images,
                    ema_decay=args.ema_decay,
                    rank=rank,
                    distributed=distributed
                )
                # if dataset == "celebA":
                #     dataSET = celeba

                evaluator = Evaluator(dataset=dataset, device=device) 
                # in case of elastic launch, resume should always be turned on
                
                
                
                # if args.fid:
                #     chkpt_path = "/home/damion/Code/DSD/diffusion_distill/chkpts/dist95_64_128_lsun_bedroom_128_dist_1_5000.pt"
                #     trainer.load_checkpoint(chkpt_path, map_location='cuda:0')
                #     print("FID:", args.fid)
                #     # trainer.generate_imgs()
                    

                # use cudnn benchmarking algorithm to select the best conv algorithm
                if torch.backends.cudnn.is_available():  # noqa
                    torch.backends.cudnn.benchmark = True  # noqa
                

                fid = trainer.FID(evaluator, name)
                new_row = pd.DataFrame({
                'dataset': [dataset],
                'checkpoint': [chkpt],
                'fid': [fid]})
            except:
                new_row = pd.DataFrame({
                'dataset': [dataset],
                'checkpoint': [chkpt],
                'fid': ["FAIL"]})

            # Append the new data
            fid_dataframe = pd.concat([fid_dataframe, new_row])
            
            
            fid_dataframe.to_csv("FIDs.csv")

def wandb_log(name, lr, tags, notes, project="cvpr_Diffusion"):
    """
    Params: wandb name, lr, model, wand tags, wandb notes. Task: returns a wandb session with CIFAR-1000 information,
    logs: Loss, Generational Loss, hardware specs, model gradients
    """
    session = wandb.init(
    project=project, 
    name=name, 
    config={"learning_rate": lr, "architecture": "Diffusion Model","dataset": "Imagenet-1000"}, tags=tags, notes=notes)
    # session.watch(model, log="all", log_freq=1000)
    return session

if __name__ == "__main__":
    from argparse import ArgumentParser


    


    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "celeba", "lsun", "lsun_bedroom"], default="cifar10")
    parser.add_argument("--root", default="./datasets", type=str, help="root directory of datasets")
    parser.add_argument("--epochs", default=120, type=int, help="total number of training epochs")
    parser.add_argument("--lr", default=0.0002, type=float, help="learning rate")
    parser.add_argument("--beta1", default=0.9, type=float, help="beta_1 in Adam")
    parser.add_argument("--beta2", default=0.999, type=float, help="beta_2 in Adam")
    parser.add_argument("--weight-decay", default=0., type=float,
                        help="decoupled weight_decay factor in Adam")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--num-accum", default=10, type=int, help=(
        "number of batches before weight update, a.k.a. gradient accumulation"))
    parser.add_argument("--train-timesteps", default=128, type=int, help=(
        "number of diffusion steps for training (0 indicates continuous training)"))
    parser.add_argument("--sample-timesteps", default=128, type=int, help="number of diffusion steps for sampling")
    parser.add_argument("--logsnr-schedule", choices=["linear", "sigmoid", "cosine", "legacy"], default="cosine")
    parser.add_argument("--logsnr-max", default=20., type=float)
    parser.add_argument("--logsnr-min", default=-20., type=float)
    parser.add_argument("--model-out-type", choices=["x_0", "eps", "both", "v"], default="v", type=str)
    parser.add_argument("--model-var-type", choices=["fixed_small", "fixed_large", "fixed_medium"], default="fixed_large", type=str)
    parser.add_argument("--reweight-type", choices=["constant", "snr", "truncated_snr", "alpha2"], default="truncated_snr", type=str)
    parser.add_argument("--loss-type", choices=["kl", "mse"], default="mse", type=str)
    parser.add_argument("--intp-frac", default=0., type=float)
    parser.add_argument("--use-cfg", action="store_true", help="whether to use classifier-free guidance")
    parser.add_argument("--w-guide", default=0.1, type=float, help="classifier-free guidance strength")
    parser.add_argument("--p-uncond", default=0.1, type=float, help="probability of unconditional training")
    parser.add_argument("--num-workers", default=4, type=int, help="number of workers for data loading")
    parser.add_argument("--train-device", default="cuda:0", type=str)
    parser.add_argument("--eval-device", default="cuda:0", type=str)
    parser.add_argument("--image-dir", default="./images/train", type=str)
    parser.add_argument("--image-intv", default=1, type=int)
    parser.add_argument("--num-save-images", default=5, type=int, help="number of images to generate & save")
    parser.add_argument("--sample-bsz", default=5, type=int, help="batch size for sampling")
    parser.add_argument("--config-dir", default="./configs", type=str)
    parser.add_argument("--chkpt-dir", default="./chkpts", type=str)
    parser.add_argument("--chkpt-name", default="", type=str)
    parser.add_argument("--chkpt-intv", default=1, type=int, help="frequency of saving a checkpoint")
    parser.add_argument("--seed", default=1234, type=int, help="random seed")
    parser.add_argument("--resume", action="store_true", help="to resume training from a checkpoint")
    parser.add_argument("--eval", action="store_true", help="whether to evaluate fid during training")
    parser.add_argument("--use-ema", action="store_true", help="whether to use exponential moving average")
    parser.add_argument("--use-ddim", action="store_true", help="whether to use DDIM sampler")
    parser.add_argument("--ema-decay", default=0.9999, type=float, help="decay factor of ema")
    parser.add_argument("--distributed", action="store_true", help="whether to use distributed training")
    parser.add_argument("--distill", action="store_true", help="whether to distillation during training")
    parser.add_argument("--name", type=str, help="wandb name", default="TD_lsun")
    parser.add_argument("--fid", action="store_true", help="generate images for FID")
    args = parser.parse_args()

    # session = wandb_log(name=args.name, lr=args.lr, tags=["train_distill", args.dataset], notes="", project="train_distill")
    session =None
    # args.session = session
    main(args)

# python FID.py --fid_imgs 5000
import os
import re
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from .utils import save_image, EMA, wandb_image
from .metrics.fid_score import InceptionStatistics, get_precomputed, calc_fd
from tqdm import tqdm
from contextlib import nullcontext
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torchvision.utils import make_grid
import wandb
from PIL import Image
import numpy as np

class DummyScheduler:
    @staticmethod
    def step():
        pass

    def load_state_dict(self, state_dict):
        pass

    @staticmethod
    def state_dict():
        return None


class RunningStatistics:
    def __init__(self, **kwargs):
        self.count = 0
        self.stats = []
        for k, v in kwargs.items():
            self.stats.append((k, v or 0))
        self.stats = dict(self.stats)

    def reset(self):
        self.count = 0
        for k in self.stats:
            self.stats[k] = 0

    def update(self, n, **kwargs):
        self.count += n
        for k, v in kwargs.items():
            self.stats[k] = self.stats.get(k, 0) + v

    def extract(self):
        avg_stats = []
        for k, v in self.stats.items():
            avg_stats.append((k, v/self.count))
        return dict(avg_stats)

    def __repr__(self):
        out_str = "Count(s): {}\n"
        out_str += "Statistics:\n"
        for k in self.stats:
            out_str += f"\t{k} = {{{k}}}\n"  # double curly-bracket to escape
        return out_str.format(self.count, **self.stats)


class Trainer:
    def __init__(
            self,
            model,
            optimizer,
            distill_optimizer,
            diffusion,
            timesteps,
            epochs,
            trainloader,
            sampler=None,
            scheduler=None,
            use_cfg=False,
            use_ema=False,
            grad_norm=1.0,
            num_accum=1,
            shape=None,
            device=torch.device("cpu"),
            chkpt_intv=5,  # save a checkpoint every [chkpt_intv] epochs
            image_intv=1,  # generate images every [image_intv] epochs
            num_save_images=64,
            ema_decay=0.9999,
            distributed=False,
            rank=0  # process id for distributed training
    ):
        self.model = model
        self.half_timesteps = int(timesteps / 2)
        self.distill_optimizer = distill_optimizer
        self.optimizer = optimizer
        self.diffusion = diffusion
        self.timesteps = timesteps
        self.epochs = epochs
        self.start_epoch = 0
        self.trainloader = trainloader
        self.sampler = sampler
        if shape is None:
            shape = next(iter(trainloader))[0].shape[1:]
        self.shape = shape
        self.scheduler = DummyScheduler() if scheduler is None else scheduler

        self.grad_norm = grad_norm
        self.num_accum = num_accum
        self.device = device
        self.chkpt_intv = chkpt_intv
        self.image_intv = image_intv
        self.num_save_images = num_save_images

        if distributed:
            assert sampler is not None
        self.distributed = distributed
        self.is_main = rank == 0

        self.use_cfg = use_cfg
        self.use_ema = use_ema
        if self.is_main and use_ema:
            self.ema = EMA(self.model, decay=ema_decay)
        else:
            self.ema = nullcontext()

        self.stats = RunningStatistics(loss=None)

    def loss(self, x, y):
        B = x.shape[0]
        T = self.timesteps
        if T > 0:
            t = torch.randint(
                T, size=(B, ), dtype=torch.float32, device=self.device
            ).add(1).div(self.timesteps)
        else:
            t = torch.rand((B, ), dtype=torch.float32, device=self.device)
        loss, predict = self.diffusion.train_losses(self.model, x_0=x, t=t, y=y)
        assert loss.shape == (B, )
        return loss
    
    def train_distill_loss(self, x, y, specified_t=None):
        B = x.shape[0]
        T = self.timesteps

        if T > 0:
            t_raw = torch.randint(T-1, size=(B,), dtype=torch.float32, device=self.device).add(1)
    
            # The related t_raw_incremented is just t_raw + 1
            t_raw_incremented = t_raw.add(1)
            # print(t_raw)
            # print(t_raw_incremented)
            # Then divide both by self.timesteps to get them into the original scale
            t = t_raw.div(self.timesteps)
            t_incremented = t_raw_incremented.div(self.timesteps)
     
            
        else:
            t = torch.rand((B, ), dtype=torch.float32, device=self.device)
            t_incremented = None  # You can choose to handle this differently if needed

            
        main_loss, predict = self.diffusion.train_losses(self.model, x_0=x, t=t, y=y)
        distill_loss = self.diffusion.distill_loss(x_0=x, t=t_incremented, y=y, denoise_fn=self.model, predict=predict)
        
        
        assert main_loss.shape == (B, )
        return main_loss.mean(), distill_loss.mean()
    
    def step(self, x, y, update=True):
        B = x.shape[0]
        loss = self.loss(x, y).mean()
        loss.div(self.num_accum).backward()
        if update:
            # gradient clipping by global norm
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            # adjust learning rate every step (warming up)
            self.scheduler.step()
            if self.is_main and self.use_ema:
                self.ema.update()
        self.stats.update(B, loss=loss.item() * B)
        return loss.item()

    def step_distill(self, x, y, update=True, session=None, i=0, distill_t = None):
        B = x.shape[0]
        main_loss, distill_loss = self.train_distill_loss(x, y)

        
        # Combine losses with weights (hyperparameters)
        alpha = 0.9  # weight for main loss
        beta = 0.1  # weight for distillation loss
        
        loss = alpha * main_loss + beta * distill_loss
        # loss = main_loss
        
        loss.div(self.num_accum).backward()
        if update:
            # gradient clipping by global norm
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            # adjust learning rate every step (warming up)
            self.scheduler.step()
            if self.is_main and self.use_ema:
                self.ema.update()
        self.stats.update(B, loss=loss.item() * B)
        
        
        return loss.item(), main_loss.item(), distill_loss.item()

    def sample_fn(
            self, noises, labels,
            diffusion=None, use_ddim=False, batch_size=-1, timesteps=None, half = False):
        
        if diffusion is None:
            diffusion = self.diffusion
        
        if timesteps == None:
            if half:
                timesteps = self.half_timesteps 
                print("setting half timesteps")
            diffusion.sample_timesteps = timesteps
            diffusion.timesteps = timesteps
        shape = noises.shape
        with self.ema:
            if batch_size == -1:
                sample = diffusion.p_sample(
                    denoise_fn=self.model, shape=shape, device=self.device,
                    noise=noises, label=labels, use_ddim=use_ddim, timesteps=timesteps)
            else:
                sample = []
                for i in range(0, noises.shape[0], batch_size):
                    print("no ema, timesteps:", timesteps)
                    _slice = slice(i, i+batch_size)
                    _shape = (min(noises.shape[0] - i, batch_size), ) + tuple(shape[1:])
                    sample.append(diffusion.p_sample(
                        denoise_fn=self.model, shape=_shape, device=self.device,
                        noise=noises[_slice], label=labels[_slice], use_ddim=use_ddim, timesteps=timesteps))
                sample = torch.cat(sample, dim=0)
        assert sample.grad is None
        return sample

    def random_labels(self):
        if self.multitags:
            inds = torch.randint(
                len(self.trainloader.dataset),
                size=(self.num_save_images,))
            labels = torch.as_tensor(
                self.trainloader.dataset.targets[inds], dtype=torch.float32)
        else:
            labels = torch.arange(self.num_classes, dtype=torch.float32) + 1
            repeats = torch.as_tensor([
                (self.num_save_images // self.num_classes
                 + int(i < self.num_save_images % self.num_classes))
                for i in range(self.num_classes)])
            labels = labels.repeat_interleave(repeats)
        return labels

    @property
    def num_classes(self):
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module.num_classes
        else:
            return self.model.num_classes

    @property
    def multitags(self):
        if isinstance(self.model, DistributedDataParallel):
            return self.model.module.multitags
        else:
            return self.model.multitags

    def train(
            self,
            evaluator=None,
            noises=None,
            labels=None,
            chkpt_path=None,
            image_dir=None,
            use_ddim=False,
            sample_bsz=-1,
            session=None,
            distill=False,
            distill_optimizer=None,
            timesteps=128,
            fid=False,
            name="test"
    ):

        if self.is_main and self.num_save_images:
            if noises is None:
                # fixed x_T for image generation
                noises = torch.randn((self.num_save_images, ) + self.shape)
            if labels is None and self.num_classes:
                labels = self.random_labels()

        total_batches = 0
        for e in range(self.start_epoch, self.epochs):
            self.stats.reset()
            self.model.train()
            if isinstance(self.sampler, DistributedSampler):
                self.sampler.set_epoch(e)
            with tqdm(
                    self.trainloader,
                    desc=f"{e+1}/{self.epochs} epochs",
                    disable=not self.is_main
            ) as t:
                for i, (x, y) in enumerate(t):
                    total_batches += 1
                    if not self.use_cfg:
                        y = None
                    

                    if distill:
                        combined_loss, train_loss, distill_loss = self.step_distill(
                            x.to(self.device),
                            y.float().to(self.device)
                            if y is not None else y,
                            update=total_batches % self.num_accum == 0, session=session, i=i
                        )

                        if session != None:
                            # session.log({"process_loss": self.current_stats["loss"]})
                            session.log({"combined_loss": combined_loss})
                            session.log({"train_loss": train_loss})
                            session.log({"distill_loss": distill_loss})
                    else:
                        loss = self.step(
                            x.to(self.device),
                            y.float().to(self.device)
                            if y is not None else y,
                            update=total_batches % self.num_accum == 0
                            )
                        if session != None:
                            # session.log({"process_loss": self.current_stats["loss"]})
                            session.log({"train_loss": loss})
                           
                        
                    t.set_postfix(self.current_stats)

                    
                    
                    if i == len(self.trainloader) - 1:
                        self.model.eval()
                        if evaluator is not None:
                            eval_results, fid = evaluator.eval(self.sample_fn, noises=noises, labels=labels, max_eval_count=500)
                            session.log({"final fid (2000)": fid})
                        else:
                            eval_results = dict()
                        results = dict()
                        results.update(self.current_stats)
                        results.update(eval_results)
                        t.set_postfix(results)
               
                    
                    if session != None and total_batches % 5000 == 0:
                    # if total_batches % 5000 == 0:
                   
                        
                        self.model.eval()
                        x_first = self.sample_fn(
                        noises=noises, labels=labels, use_ddim=use_ddim, batch_size=sample_bsz, timesteps=timesteps, half=False)
                        wandb_image(x_first, f"{timesteps}")
                        
                        x = self.sample_fn(
                        noises=noises, labels=labels, use_ddim=use_ddim, batch_size=sample_bsz, timesteps=self.half_timesteps, half=True)
                        wandb_image(x, f"{int(timesteps / 2)}")
                        # save_image(x, os.path.join(image_dir, f"{e+1}.jpg"), session=session)
                        # print(x == x_first)

                        # # if evaluator is not None:
                        # base_folder = "./FID_IMAGES/"
                        # filename = name
                        # # Combine them to get the full path
                        # full_path = os.path.join(base_folder, filename)

                        # # Extract the folder path from the full path
                        # folder_path = os.path.dirname(full_path)
                        # if not os.path.exists(folder_path):
                        #     os.makedirs(folder_path)

                        # eval_results, fid = evaluator.eval(self.sample_fn, noises=noises, labels=labels, max_eval_count=50, timesteps = timesteps, folder_path=folder_path)
                        # # eval_results, fid = evaluator.eval(self.sample_fn, noises=noises, labels=labels, max_eval_count=100, timesteps = timesteps, folder_path=folder_path)
                        # session.log({f"fid {timesteps}": fid})
                        # new_timesteps = int(timesteps / 2)
                        # eval_results, fid = evaluator.eval(self.sample_fn, noises=noises, labels=labels, max_eval_count=50, timesteps = new_timesteps, folder_path=folder_path)
                        # session.log({f"fid {new_timesteps}": fid})
                        

                        torch.save(self.model, f'./chkpts/{name}_{total_batches}.pth')
                        self.model.train()

                        
                     
                      
                        # try:
                        
                        # except e:
                        #     print("FID FAILED")
                        #     print("E:", e)
                        #     continue
                        
                        

                    # if session != None and i % 100 == 0:
                            
                        
                    # if distill_t != None and i > 100 and i % distill_t == 0:
                    #     for j in range(1, self.timesteps + 1, 2):
                    #         loss = self.diffusion.distill(
                    #             x.to(self.device),
                    #             y.float().to(self.device)
                    #             if y is not None else y,
                    #             update=total_batches % self.num_accum == 0, session=session, distill_t=distill_t
                    #         , denoise_fn = self.model, timesteps=self.timesteps, i=j)
                    #         loss.backward()
                    #         # loss.div(self.num_accum).backward()

                    #         nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm)
                    #         self.distill_optimizer.step()
                    #         self.distill_optimizer.zero_grad(set_to_none=True)
                    #         if session != None:
                    #             session.log({"distill_loss": loss.item()})
                    #         # if self.is_main and self.use_ema:
                    #         #     self.ema.update()
                        
                        
            if self.is_main:
                if not (e + 1) % self.image_intv and self.num_save_images and image_dir:
                    # x = self.sample_fn(
                    #     noises=noises, labels=labels, use_ddim=use_ddim, batch_size=sample_bsz)
                    if session != None:
                        x = self.sample_fn(
                        noises=noises, labels=labels, use_ddim=use_ddim, batch_size=sample_bsz, timesteps=timesteps)
                        wandb_image(x, f"{timesteps}")
                        x = self.sample_fn(
                        noises=noises, labels=labels, use_ddim=use_ddim, batch_size=sample_bsz, timesteps=int(timesteps / 2))
                        wandb_image(x, f"{int(timesteps / 2)}")
                    # save_image(x, os.path.join(image_dir, f"{e+1}.jpg"), session=session)
                # if not (e + 1) % self.chkpt_intv and chkpt_path:
                #     self.save_checkpoint(chkpt_path, epoch=e+1, **results)
            if self.distributed:
                dist.barrier()  # synchronize all processes here


    def generate_imgs(
            self,
            session=None,
            timesteps=[128]
    ):
        """
        Does not work yet, chkpts not including all the layers
        """
        import os

        chkpt_path = './chkpts/FID/'
        FID_path = './FID_IMAGES/'
        files = []
        fid_paths = []

        # Check if the folder exists
        if os.path.exists(chkpt_path):
            # List all files in the directory
            for filename in os.listdir(chkpt_path):
                
                file_path = os.path.join(chkpt_path, filename) 
                fid_path = os.path.join(FID_path, filename) + "/"
                # Check if it's a file (and not a directory)
                if os.path.isfile(file_path):
                    files.append(filename)
                    fid_paths.append(fid_path)
        print(files)
        for i, path in enumerate(files):
            self.load_checkpoint(chkpt_path + path, map_location="cuda:0")
            with tqdm.tqdm(range(50)) as image_generator:
                              
                self.model.eval()
                self.num_save_images = 1
                self.num_classes = 1
                self.sample_bsz = 1


                for j, e in enumerate(image_generator):

                    if not self.use_cfg:
                        y = None
                    
                    noises = torch.randn((1)  + self.shape)    
                    labels = self.random_labels()

                    x = self.sample_fn(
                        noises=noises, labels=labels, use_ddim=True)      
                            
                    save_image(x, os.path.join(fid_folder, f"{j}.jpg"), session=session)        


                    

  

    @property
    def trainees(self):
        roster = ["model", "optimizer"]
        if self.use_ema:
            roster.append("ema")
        if self.scheduler is not None:
            roster.append("scheduler")
        return roster

    @property
    def current_stats(self):
        return self.stats.extract()

    def load_checkpoint(self, chkpt_path, map_location):
        chkpt = torch.load(chkpt_path, map_location=map_location)
        for trainee in self.trainees:
            try:
                getattr(self, trainee).load_state_dict(chkpt[trainee])
            except RuntimeError:
                _chkpt = chkpt[trainee]["shadow"] if trainee == "ema" else chkpt[trainee]
                for k in list(_chkpt.keys()):
                    if k.split(".")[0] == "module":
                        _chkpt[".".join(k.split(".")[1:])] = _chkpt.pop(k)
                getattr(self, trainee).load_state_dict(chkpt[trainee])
            except AttributeError:
                continue
        self.start_epoch = chkpt["epoch"]

    def save_checkpoint(self, chkpt_path, **extra_info):
        chkpt = []
        for k, v in self.named_state_dicts():
            chkpt.append((k, v))
        for k, v in extra_info.items():
            chkpt.append((k, v))
        if "epoch" in extra_info:
            chkpt_path = re.sub(r"(_\d+)?\.pt", f"_{extra_info['epoch']}.pt", chkpt_path)
        torch.save(dict(chkpt), chkpt_path)

    def named_state_dicts(self):
        for k in self.trainees:
            yield k, getattr(self, k).state_dict()


class Evaluator:
    def __init__(
            self,
            dataset,
            diffusion=None,
            eval_batch_size=256,
            max_eval_count=1000,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.diffusion = diffusion
        # inception stats
        self.istats = InceptionStatistics(device=device)
        self.eval_batch_size = eval_batch_size
        self.max_eval_count = max_eval_count
        self.device = device
        self.target_mean, self.target_var = get_precomputed(dataset)

    def eval(self, sample_fn, noises, labels, max_eval_count=None, timesteps=128, folder_path=None):
        if max_eval_count == None:
            max_eval_count = self.max_eval_count
        if noises is None:
                # fixed x_T for image generation
                noises = torch.randn((self.num_save_images, ) + self.shape)
        if labels is None and self.num_classes:
                labels = self.random_labels()
        self.istats.reset()
        for index in range(0, max_eval_count + self.eval_batch_size, self.eval_batch_size):
            x = sample_fn(
                        noises=noises, labels=labels, use_ddim=True, timesteps=timesteps)
            if folder_path != None:
                save_image(x, os.path.join(folder_path, f"{timesteps}_{index +1}.jpg"))
            self.istats(x.to(self.device))
        gen_mean, gen_var = self.istats.get_statistics()
        fid = calc_fd(gen_mean, gen_var, self.target_mean, self.target_var)
        return {"fid": fid}, fid

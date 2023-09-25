"""
This script is modified from the original PyTorch Implementation of FID
(https://github.com/mseitzer/pytorch-fid/) to support fid evaluation on
the fly without writing data onto the disk during the training process.
"""

"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from PIL import Image
from scipy import linalg
from .inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from torchvision import transforms
import io
import os
import pathlib
import numpy as np
import requests
import torch
import torch.nn as nn
import torch.utils.data

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

IMAGE_RES = 299

# Note that InceptionV3 forward method already applied this transformation
INPUT_TRANSFORM = transforms.Compose([
    transforms.Resize(
        size=(IMAGE_RES, IMAGE_RES),
        interpolation=transforms.InterpolationMode.BILINEAR  # bilinear interpolation
    ),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )  # rescale [0, 1] to [-1, 1]
])


class InceptionStatistics(nn.Module):
    def __init__(
            self,
            model=None,
            input_transform=lambda x: x,
            activation_dim=2048,
            device=torch.device("cpu")
    ):
        super(InceptionStatistics, self).__init__()
        self.input_transform = input_transform
        self.activation_dim = activation_dim
        self.device = device
        if model is None:
            self.model = self.load_model()
        else:
            self.model = model
        self.model.eval()
        self.model.to(device)
        # initialize statistics
        self.running_mean = np.zeros((activation_dim,), dtype=np.float64)
        self.running_var = np.zeros((activation_dim, activation_dim), dtype=np.float64)
        self.count = 0

    def load_model(self):
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.activation_dim]
        model = InceptionV3([block_idx])
        return model

    def forward(self, x):
        x = self.input_transform(x)
        with torch.inference_mode():
            act = self.model(x)[0].cpu().numpy()
        if act.shape[2] != 1 or act.shape[3] != 1:
            act = adaptive_avg_pool2d(act, (1, 1))
        act = act.squeeze(-1).squeeze(-1)
        mean = np.mean(act, axis=0, dtype=np.float64)
        var = np.cov(act, rowvar=False, ddof=0, dtype=np.float64)
        count = act.shape[0]
        alpha = count / (self.count + count)
        if self.count == 0:
            self.running_mean += mean
            self.running_var += var
        else:
            mean_diff = (mean - self.running_mean)
            self.running_mean += alpha * mean_diff
            self.running_var += alpha * (var - self.running_var)
            self.running_var += alpha * (1 - alpha) * np.outer(mean_diff, mean_diff)
        self.count += count

    def get_statistics(self):
        assert self.count > 1, "Count must be greater than 1!"
        return (
            self.running_mean,
            self.running_var * self.count / (self.count - 1)
        )

    def reset(self):
        self.running_mean.fill(0)
        self.running_var.fill(0)
        self.count = 0


PRE_COMPUTED_LIST = {
    # "cropped_celeba": "http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_celeba.npz",
    "cropped_celeba": "https://github.com/tqch/VAEGAN/releases/download/precomputed_statistics_celeba/fid_stats_celeba_148x148.npz",
    "lsun": "http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_lsun_church_64.npz",
    "lsun_bedroom": "http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_lsun_train.npz",
    "cifar10": "http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz",
    "svhn": "http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_svhn_train.npz",
    "imagenet_train": "http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_imagenet_train.npz",
    "imagenet_valid": "http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_imagenet_valid.npz"
}


def get_precomputed(dataset, download_dir="precomputed"):
    if dataset == "celeba":
        dataset = "cropped_celeba"
    url = PRE_COMPUTED_LIST[dataset]
    filename = os.path.basename(url)
    if download_dir is None:
        file_obj = io.BytesIO(requests.get(url).content)
    else:
        filepath = os.path.join(download_dir, filename)
        if not os.path.exists(filepath):
            try:
                os.makedirs(download_dir)
            except FileExistsError:
                pass
            r = requests.get(url)
            with open(filepath, "wb") as file:
                file.write(r.content)
            file_obj = io.BytesIO(r.content)
        else:
            file_obj = filepath
    precomputed_data = np.load(file_obj)
    mean = precomputed_data["mu"]
    var = precomputed_data["sigma"]
    return mean, var


def calc_fd(mean1, var1, mean2, var2, eps=1e-6):
    return calculate_frechet_distance(mean1, var1, mean2, var2, eps)


###############################################################
# Original code below
###############################################################


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(files, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    dataset = ImagePathDataset(files, transforms=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=num_workers)

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- device      : Device to run calculations
    -- num_workers : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, dims, device,
                               num_workers=1):
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                        for file in path.glob('*.{}'.format(ext))])
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, device, num_workers)

    return m, s


def calculate_fid_given_paths(paths, batch_size, device, dims, num_workers=1):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size,
                                        dims, device, num_workers)
    m2, s2 = compute_statistics_of_path(paths[1], model, batch_size,
                                        dims, device, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def main():
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = args.num_workers

    fid_value = calculate_fid_given_paths(args.path,
                                          args.batch_size,
                                          device,
                                          args.dims,
                                          num_workers)
    print('FID: ', fid_value)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size to use')
    parser.add_argument('--num-workers', type=int,
                        help=('Number of processes to use for data loading. '
                              'Defaults to `min(8, num_cpus)`'))
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--dims', type=int, default=2048,
                        choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))
    parser.add_argument('path', type=str, nargs=2,
                        help=('Paths to the generated images or '
                              'to .npz statistic files'))

    IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                        'tif', 'tiff', 'webp'}
    main()

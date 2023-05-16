"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from randaugment import RandomAugment
import lavis.tasks as tasks
from lavis.models import load_preprocess, load_model_and_preprocess
from lavis.common.config import Config
from lavis.infer_utils import get_images_texts, infer
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    # job_id = now()
    #
    # cfg = Config(parse_args())
    #
    # init_distributed_mode(cfg.run_cfg)
    #
    # setup_seeds(cfg)
    #
    # # set after init_distributed_mode() to only log on master.
    # setup_logger()
    #
    # cfg.pretty_print()
    # task = tasks.setup_task(cfg)
    # # datasets = task.build_datasets(cfg)
    # # model = task.build_model(cfg)

    # normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    #
    # transform_train = transforms.Compose([
    #     transforms.RandomResizedCrop(224, scale=(0.5, 1.0),
    #                                  interpolation=InterpolationMode.BICUBIC),
    #     transforms.RandomHorizontalFlip(),
    #     RandomAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
    #                                           'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
    #     transforms.ToTensor(),
    #     normalize,
    # ])


    images_root = '/sda/home/lipeng/Program/BLIP/data/datasets/test/test_images'
    texts_root = '/sda/home/lipeng/Program/BLIP/data/datasets/test/test_text.txt'

    name = "pretrain_vitL"#"coco"#"flickr" # "pretrain_vitL"
    model_name = "blip2_image_text_matching"#"albef_retrieval"#"blip2_image_text_matching"
    print(model_name, name)
    model, vis_processors, text_processors = load_model_and_preprocess(model_name, name,
                                                                       device=device, is_eval=True)

    images, texts, idx_dic = get_images_texts(images_root, texts_root, vis_processors)
    sim_matrix = model.get_sim(zip(images, idx_dic.keys()), texts, k=128)
    infer(sim_matrix, idx_dic, texts, name=f'{model_name}_{name}')
    print(model_name, name)



if __name__ == "__main__":
    main()

"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

import lavis.tasks as tasks
from lavis.models import load_preprocess, load_model_and_preprocess
from lavis.common.config import Config
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
    job_id = now()

    cfg = Config(parse_args())

    init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    # datasets = task.build_datasets(cfg)
    # model = task.build_model(cfg)
    raw_image = Image.open("./docs/_static/merlion.png").convert("RGB")
    caption = "merlion in Singapore"
    model, vis_processors, text_processors = load_model_and_preprocess("blip2_opt", "caption_coco_opt2.7b", device=device, is_eval=True)
    print(model)
    img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    txt = text_processors["eval"](caption).split(' ')
    # output = model.generate({"image": img})
    itm_output = model({"image": img, "text_input": txt})
    # runner = RunnerBase(
    #     cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    # )
    # runner.evaluate(skip_reload=True)
    # model.to(device)
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    #
    # inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    #
    # generated_ids = model.generate(**inputs)
    # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    # print(generated_text)

if __name__ == "__main__":
    main()

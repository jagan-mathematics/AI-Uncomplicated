from dataclasses import field
import json
import os
from pathlib import Path
import re
from typing import List, Optional, Tuple
from omegaconf import OmegaConf
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh
import torch

import torch.nn as nn

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict
)



import logging

logger = logging.getLogger(__name__)

# todo: did some changes in name, have to verify
DUMP_FOLDER_NAME = "checkpoint_{:010d}" # Formatted as ten digit number padded with zeros in front (eg., 0000000001)
EVAL_FOLDER_NAME = "eval_{:010d}"
RE_FOLDER = r"(?:checkpoint|eval)_\d{10}"

CONSOLIDATE_FOLDER = "consolidated"
CONSOLIDATE_NAME = "consolidated.pth"

CONFIG_NAME = "params.json"
TRAIN_STATE_NAME = "train_state_{:05d}.json"


RE_DIGIT = r"\d+"


class SaveEvery:
    steps: int = 1000
    limit: int = 0


class CheckpointArgs:
    path: Optional[str] = None
    save_every: int = field(default_factory=SaveEvery)
    eval_every: int = field(default_factory=SaveEvery)
    init_checkpoint_path: Optional[str] = None
    continue_training_from_init: bool = False


def _get_key_step(name: str):
    return int(re.findall(RE_DIGIT, name)[-1])


class CheckpointManager:
    def __init__(self, args: CheckpointArgs):
        self.path = args.path
        self.save_every = args.save_every
        self.eval_every = args.eval_every
        self.init_ckpt_path = args.init_ckpt_path
        self.continue_training_from_init = args.continue_training_from_init

        assert self.path and os.path.exists(self.path), f"Path {self.path} does not exist and needs to be created before using CheckpointManager (use instantiate_and_make_dir)"

        self.existing_saves = self.get_existing_saves()

    def get_existing_saves(self) -> List[Path]:
        folders = [
            p
            for p in Path(self.path).iterdir()
            if p.is_dir() and re.match(RE_FOLDER, p.name)
        ]
        folders.sort(key=lambda p: _get_key_step(p.name))
        return folders

    def clean_up(self):
        logger.info("Cleaning up checkpoints...")
        dump_folders = []
        eval_folders = []
        other_folders = []
        for p in self.existing_saves:
            is_dump = p.name.startswith("checkpoint")
            is_eval = p.name.startswith("eval")
            if is_dump:
                dump_folders.append(p)
            if is_eval:
                eval_folders.append(p)
            if not (is_dump or is_eval):
                other_folders.append(p)

        logger.info(f"Dump folders: {dump_folders}")
        logger.info(f"Eval folders: {eval_folders}")
        logger.info(f"Other folders: {other_folders}")

        if self.save_every.limit > 0:
            dump_folders = dump_folders[-self.save_every.limit :]
        if self.eval_every.limit > 0:
            eval_folders = eval_folders[-self.eval_every.limit :]

        folder_to_keep = set(other_folders + dump_folders + eval_folders)
        folder_to_remove = set(self.existing_saves) - folder_to_keep

        logger.info(f"Removing folders: {folder_to_remove}")

        if dist.get_rank() == 0:
            for folder in folder_to_remove:
                for file in folder.iterdir():
                    if file.is_file():
                        file.unlink()
                    elif file.is_dir():
                        assert file.name in [CONSOLIDATE_FOLDER]
                        for f in file.iterdir():
                            f.unlink()
                        file.rmdir()
                folder.rmdir()

        dist.barrier()

        self.existing_saves = list(folder_to_keep)
        self.existing_saves.sort(key=lambda p: _get_key_step(p.name))


    @classmethod
    def instantiate_and_make_dir(cls, args: CheckpointArgs):
        if get_is_master(): # todo: have to implement in dist folder
            os.makedirs(args.path, exist_ok=True)
        dist.barrier()

        return cls(args)

    def _create_folder(self, base_path: Path, folder_name: str) -> Path:
        folder = base_path / folder_name
        if get_is_master(): # todo: have to implement in dist folder
            folder.mkdir(parents=False, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()
        return folder


    def _get_dp_tp_mesh(
        self, device_mesh: Optional[DeviceMesh] = None
    ) -> Tuple[int, int]:
        dp_rank = 0
        tp_rank = 0
        if device_mesh is not None:
            if "dp_replicate" in device_mesh.mesh_dim_names:
                dp_rank = device_mesh.get_local_rank("dp_replicate")
                if "dp_shard" in device_mesh.mesh_dim_names:
                    dp_rank = dp_rank * device_mesh["dp_shard"].size() + device_mesh.get_local_rank("dp_shard")
            if "tp" in device_mesh.mesh_dim_names:
                tp_rank = device_mesh.get_local_rank("tp")
        return dp_rank, tp_rank


    @torch.no_grad()
    def get_state_dict(
        self,
        model,
        optimizer,
    ):
        model_sd, optim_sd = get_state_dict(model, optimizer)
        return {"model": model_sd, "optim": optim_sd}


    def save(
        self,
        model,
        optimizer,
        train_state,
        config,
        device_mesh: Optional[DeviceMesh] = None,
    ) -> bool:

        # When creating directory check if only rank0 or is there other solution
        path = Path(self.path)
        curr_save_dir = self._create_folder(path, DUMP_FOLDER_NAME.format(train_state.step))
        logger.info(f"Saving to: {str(curr_save_dir)}")

        if dist.is_initialized():
            dist.barrier()

        logger.info("Saving...")
        state_dict = self.get_state_dict(model, optimizer)
        dcp.save(state_dict, checkpoint_id=curr_save_dir)
        logger.info("State dict saved!")

        if dist.is_initialized():
            dist.barrier()

        if get_is_master(): # todo: have to implement in dist folder
            with open(curr_save_dir / CONFIG_NAME, "w") as f:
                json.dump(
                    OmegaConf.to_container(OmegaConf.structured(config), resolve=True),
                    f,
                )

        # Add json dump here
        dp_rank, tp_rank = self._get_dp_tp_mesh(device_mesh)
        if tp_rank == 0:
            train_state_name = TRAIN_STATE_NAME.format(dp_rank)
            logger.info(
                f"Saving train state to: {str(curr_save_dir / train_state_name)}"
            )
            with open(curr_save_dir / train_state_name, "w") as f:
                json.dump(train_state.state_dict(), f)
            logger.info("Train state saved !")

        self.existing_saves.append(curr_save_dir)

        self.clean_up()

        if dist.is_initialized():
            dist.barrier()
        return True


    def get_last_step_path(self, dp_rank: int = 0) -> Optional[Path]:
        path = None
        for p in reversed(self.existing_saves):
            if (p / TRAIN_STATE_NAME.format(dp_rank)).is_file():
                path = p
                break
        return path


    @torch.no_grad()
    def load(
        self,
        model: nn.Module,
        optimizer,
        train_state,
        device_mesh: DeviceMesh,
        path: Optional[Path] = None,
    ):
        dp_rank, tp_rank = self._get_dp_tp_mesh(device_mesh)
        # Loading tries to load the provided path, if not available the last saved step and finally from the init path
        path = path or self.get_last_step_path(dp_rank=dp_rank)
        # If none of those are available don't do anything
        if path is None:
            # If no checkpoints exist do nothing
            return

        # Only load train state if it's provided, the files exist and we're not loading from init path
        train_state_name = TRAIN_STATE_NAME.format(dp_rank)
        logger.info("Reloading train state")
        with open(path / train_state_name, "r") as f:
            train_state_dict = json.load(f)
        train_state.load_state_dict(train_state_dict)
        logger.info("Train state reloaded")

        logger.info(f"Loading from: {str(path)}")
        state_dict = self.get_state_dict(
            model=model,
            optimizer=optimizer,
        )
        dcp.load(state_dict, checkpoint_id=path)
        logger.info("State dict loaded.")

        logger.info("Reloading model and optim")

        set_state_dict(
            model,
            optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )
        logger.info("Model and optim reloaded")
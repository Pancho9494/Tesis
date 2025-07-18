import time, os, torch, copy
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from lib.timer import Timer, AverageMeter
from lib.utils import Logger, validate_gradient

from tqdm import tqdm
import torch.nn.functional as F
import gc
from debug.decorators import identify_method
import aim
from traceback import format_exc


class Trainer(object):
    """
    Epoch[00] Step[610] Iter[610]
    FMR average [0.1003]MultiLoss[1.3838] (Overlap(0.1609) + Matchability(0.0000) + Circle(1.2229))
    """

    def __init__(self, args):
        self.config = args
        # parameters
        self.start_epoch = 1
        self.max_epoch = args.max_epoch
        self.save_dir = args.save_dir
        self.device = args.device
        self.verbose = args.verbose
        self.max_points = args.max_points

        self.model = args.model.to(self.device)
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_freq = args.scheduler_freq
        self.snapshot_freq = args.snapshot_freq
        self.snapshot_dir = args.snapshot_dir
        self.benchmark = args.benchmark
        self.iter_size = args.iter_size
        self.verbose_freq = args.verbose_freq

        self.w_circle_loss = args.w_circle_loss
        self.w_overlap_loss = args.w_overlap_loss
        self.w_saliency_loss = args.w_saliency_loss
        self.desc_loss = args.desc_loss

        self.best_loss = 1e5
        self.best_recall = -1e5
        self.writer = SummaryWriter(log_dir=args.tboard_dir)
        self.logger = Logger(args.snapshot_dir)
        self.logger.write(f"#parameters {sum([x.nelement() for x in self.model.parameters()]) / 1000000.0} M\n")

        print(f"TOTAL NUMBER OF PARAMETERS: {sum(p.numel() for p in self.model.parameters())}")

        if args.pretrain != "":
            print(f"Loading pretrained model from {args.pretrain}")
            self._load_pretrain(args.pretrain)

        self.loader = dict()
        self.loader["train"] = args.train_loader
        self.loader["val"] = args.val_loader
        self.loader["test"] = args.test_loader

        with open(f"{args.snapshot_dir}/model", "w") as f:
            f.write(str(self.model))
        f.close()

        self.tracker = aim.Run()

    def __repr__(self) -> str:
        return "Trainer()"

    def _snapshot(self, epoch, name=None):
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "best_recall": self.best_recall,
        }
        if name is None:
            filename = os.path.join(self.save_dir, f"model_{epoch}.pth")
        else:
            filename = os.path.join(self.save_dir, f"model_{name}.pth")
        self.logger.write(f"Save model to {filename}\n")
        torch.save(state, filename)

    def _load_pretrain(self, resume):
        if os.path.isfile(resume):
            state = torch.load(resume)
            self.model.load_state_dict(state["state_dict"])
            self.start_epoch = state["epoch"]
            self.scheduler.load_state_dict(state["scheduler"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.best_loss = state["best_loss"]
            self.best_recall = state["best_recall"]

            self.logger.write(f"Successfully load pretrained model from {resume}!\n")
            self.logger.write(f"Current best loss {self.best_loss}\n")
            self.logger.write(f"Current best recall {self.best_recall}\n")
        else:
            raise ValueError(f"=> no checkpoint found at '{resume}'")

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]["lr"]

    def stats_dict(self):
        stats = dict()
        stats["c_loss"] = 0.0
        stats["circle_loss"] = 0.0
        stats["recall"] = 0.0  # feature match recall, divided by number of ground truth pairs
        stats["saliency_loss"] = 0.0
        stats["saliency_recall"] = 0.0
        stats["saliency_precision"] = 0.0
        stats["overlap_loss"] = 0.0
        stats["overlap_recall"] = 0.0
        stats["overlap_precision"] = 0.0
        return stats

    def stats_meter(self):
        meters = dict()
        stats = self.stats_dict()
        for key, _ in stats.items():
            meters[key] = AverageMeter()
        return meters

    # @identify_method
    def inference_one_batch(self, inputs, phase):
        assert phase in ["train", "val", "test"]
        ##################################
        # training
        if phase == "train":
            self.model.train()
            ###############################################
            # forward pass
            feats, scores_overlap, scores_saliency = self.model(inputs)  # [N1, C1], [N2, C2]
            pcd = inputs["points"][0]
            len_src = inputs["stack_lengths"][0][0]
            c_rot, c_trans = inputs["rot"], inputs["trans"]
            correspondence = inputs["correspondences"]

            src_pcd, tgt_pcd = inputs["src_pcd_raw"], inputs["tgt_pcd_raw"]
            src_feats, tgt_feats = feats[:len_src], feats[len_src:]

            ###################################################
            # get loss
            stats = self.desc_loss(
                src_pcd, tgt_pcd, src_feats, tgt_feats, correspondence, c_rot, c_trans, scores_overlap, scores_saliency
            )

            c_loss = (
                stats["circle_loss"] * self.w_circle_loss
                + stats["overlap_loss"] * self.w_overlap_loss
                + stats["saliency_loss"] * self.w_saliency_loss
            )

            c_loss.backward()
            stats["c_loss"] = float(c_loss.detach())

        else:
            self.model.eval()
            with torch.no_grad():
                ###############################################
                # forward pass
                feats, scores_overlap, scores_saliency = self.model(inputs)  # [N1, C1], [N2, C2]
                pcd = inputs["points"][0]
                len_src = inputs["stack_lengths"][0][0]
                c_rot, c_trans = inputs["rot"], inputs["trans"]
                correspondence = inputs["correspondences"]

                src_pcd, tgt_pcd = inputs["src_pcd_raw"], inputs["tgt_pcd_raw"]
                src_feats, tgt_feats = feats[:len_src], feats[len_src:]

                ###################################################
                # get loss
                stats = self.desc_loss(
                    src_pcd,
                    tgt_pcd,
                    src_feats,
                    tgt_feats,
                    correspondence,
                    c_rot,
                    c_trans,
                    scores_overlap,
                    scores_saliency,
                )

                c_loss = (
                    stats["circle_loss"] * self.w_circle_loss
                    + stats["overlap_loss"] * self.w_overlap_loss
                    + stats["saliency_loss"] * self.w_saliency_loss
                )
                stats["c_loss"] = float(c_loss.detach())

        ##################################
        # detach the gradients for loss terms
        stats["circle_loss"] = float(stats["circle_loss"].detach())
        stats["overlap_loss"] = float(stats["overlap_loss"].detach())
        stats["saliency_loss"] = float(stats["saliency_loss"].detach())

        return stats

    # @identify_method
    def inference_one_epoch(self, epoch, phase):
        gc.collect()
        assert phase in ["train", "val", "test"]

        # init stats meter
        stats_meter = self.stats_meter()

        num_iter = int(len(self.loader[phase].dataset) // self.loader[phase].batch_size)
        c_loader_iter = self.loader[phase].__iter__()

        self.optimizer.zero_grad()
        # for c_iter in tqdm(range(num_iter)): # loop through this epoch
        for c_iter in range(num_iter):
            ##################################
            # load inputs to device.

            print(
                f"Epoch[{epoch - 1:02d}] Step[{c_iter:02d}] Iter[{(epoch - 1) * len(self.loader[phase]) + (c_iter):02d}] ",
                end="",
            )
            inputs = c_loader_iter.next()
            for k, v in inputs.items():
                if type(v) == list:
                    inputs[k] = [item.to(self.device) for item in v]
                elif type(v) == dict:
                    pass
                else:
                    inputs[k] = v.to(self.device)
            try:
                ##################################
                # forward pass
                # with torch.autograd.detect_anomaly():
                stats = self.inference_one_batch(inputs, phase)

                ###################################################
                # run optimisation
                if (c_iter + 1) % self.iter_size == 0 and phase == "train":
                    gradient_valid = validate_gradient(self.model)
                    if gradient_valid:
                        self.optimizer.step()
                    else:
                        self.logger.write("gradient not valid\n")
                    self.optimizer.zero_grad()

                ################################
                # update to stats_meter
                # del stats["c_loss"]
                for key, value in stats.items():
                    stats_meter[key].update(value)
                    # if key == "recall":
                    #     print(stats_meter[key].val, stats_meter[key].avg)

                c_loss = (
                    stats_meter["circle_loss"].val * self.w_circle_loss
                    + stats_meter["overlap_loss"].val * self.w_overlap_loss
                    + stats_meter["saliency_loss"].val * self.w_saliency_loss
                )

                self.tracker.track(
                    stats_meter["c_loss"].avg,
                    name="MultiLoss",
                    step=c_iter,
                    epoch=epoch,
                    context={"subset": phase, "y0to1": False, "track": "average"},
                )
                self.tracker.track(
                    stats_meter["c_loss"].val,
                    name="MultiLoss",
                    step=c_iter,
                    epoch=epoch,
                    context={"subset": phase, "y0to1": False, "track": "current"},
                )
                self.tracker.track(
                    stats_meter["circle_loss"].avg * self.w_circle_loss,
                    name="CircleLoss",
                    step=c_iter,
                    epoch=epoch,
                    context={"subset": phase, "y0to1": False, "track": "average"},
                )
                self.tracker.track(
                    stats_meter["circle_loss"].val * self.w_circle_loss,
                    name="CircleLoss",
                    step=c_iter,
                    epoch=epoch,
                    context={"subset": phase, "y0to1": False, "track": "current"},
                )
                self.tracker.track(
                    stats_meter["overlap_loss"].avg * self.w_overlap_loss,
                    name="OverlapLoss",
                    step=c_iter,
                    epoch=epoch,
                    context={"subset": phase, "y0to1": False, "track": "average"},
                )
                self.tracker.track(
                    stats_meter["overlap_loss"].val * self.w_overlap_loss,
                    name="OverlapLoss",
                    step=c_iter,
                    epoch=epoch,
                    context={"subset": phase, "y0to1": False, "track": "current"},
                )
                self.tracker.track(
                    stats_meter["saliency_loss"].avg * self.w_saliency_loss,
                    name="MatchabilityLoss",
                    step=c_iter,
                    epoch=epoch,
                    context={"subset": phase, "y0to1": False, "track": "average"},
                )
                self.tracker.track(
                    stats_meter["saliency_loss"].val * self.w_saliency_loss,
                    name="MatchabilityLoss",
                    step=c_iter,
                    epoch=epoch,
                    context={"subset": phase, "y0to1": False, "track": "current"},
                )
                self.tracker.track(
                    stats_meter["recall"].avg,
                    name="FeatureMatchRecall",
                    step=c_iter,
                    epoch=epoch,
                    context={"subset": phase, "y0to1": True, "track": "average"},
                )
                self.tracker.track(
                    stats_meter["recall"].val,
                    name="FeatureMatchRecall",
                    step=c_iter,
                    epoch=epoch,
                    context={"subset": phase, "y0to1": True, "track": "current"},
                )
                p = ""
                p += "FMR "
                p += f"average [{stats_meter['recall'].avg:5.4f}]"
                p += f"MultiLoss[{c_loss:5.4f}] ("
                p += f"Overlap({stats_meter['overlap_loss'].val * self.w_overlap_loss:5.4f}) + "
                p += f"Matchability({stats_meter['saliency_loss'].val * self.w_saliency_loss:5.4f}) + "
                p += f"Circle({stats_meter['circle_loss'].val * self.w_circle_loss:5.4f})"
                p += ")"
                print(p)
            except Exception as inst:
                msg = format_exc()
                print(msg)

            torch.cuda.empty_cache()

            if (c_iter + 1) % self.verbose_freq == 0 and self.verbose:
                curr_iter = num_iter * (epoch - 1) + c_iter
                for key, value in stats_meter.items():
                    self.writer.add_scalar(f"{phase}/{key}", value.avg, curr_iter)

                message = f"{phase} Epoch: {epoch} [{c_iter + 1:4d}/{num_iter}]"
                for key, value in stats_meter.items():
                    message += f"{key}: {value.avg:.2f}\t"

                self.logger.write(message + "\n")

        message = f"{phase} Epoch: {epoch}"
        for key, value in stats_meter.items():
            message += f"{key}: {value.avg:.2f}\t"
        self.logger.write(message + "\n")

        return stats_meter

    # @identify_method
    def train(self):
        for epoch in range(self.start_epoch, self.max_epoch):
            self.inference_one_epoch(epoch, "train")
            self.scheduler.step()

            stats_meter = self.inference_one_epoch(epoch, "val")

            if stats_meter["circle_loss"].avg < self.best_loss:
                self.best_loss = stats_meter["circle_loss"].avg
                self._snapshot(epoch, "best_loss")
            if stats_meter["recall"].avg > self.best_recall:
                self.best_recall = stats_meter["recall"].avg
                self._snapshot(epoch, "best_recall")

            # we only add saliency loss when weget descent point-wise features
            print("===========================================")
            print(stats_meter["recall"])
            print(stats_meter["recall"].avg)
            if stats_meter["recall"].avg > 0.3:
                self.w_saliency_loss = 1.0
            else:
                self.w_saliency_loss = 0.0

        # finish all epoch
        print("Training finish!")

    def eval(self):
        print("Start to evaluate on validation datasets...")
        stats_meter = self.inference_one_epoch(0, "val")

        for key, value in stats_meter.items():
            print(key, value.avg)

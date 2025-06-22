import os
import torch
from src.common import compute_iou, add_key
from src.training import BaseTrainer
import aim


class Trainer(BaseTrainer):
    """
    Args:
        model (nn.Module
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        vis_dir (str): visualization directory
        threshold (float): threshold value
    """

    def __init__(self, model, optimizer, device=None, vis_dir=None, threshold=0.01):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.vis_dir = vis_dir
        self.threshold = threshold

        self.loss = torch.nn.L1Loss(reduction="none")

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

        self.tracker = aim.Run()

    def train_step(self, data, epoch_it, it):
        """Performs a training step.

        Args:
            data (dict): data dictionary
        """
        self.model.train()
        self.optimizer.zero_grad()
        output, loss = self.compute_loss(data)
        print(f"L1Loss[{loss:5.2f}]", end="")
        self.tracker.track(
            loss,
            name="L1Loss",
            step=it,
            epoch=epoch_it,
            context={"subset": "train", "y0to1": False, "track": "current"},
        )
        loss.backward()
        self.optimizer.step()

        self.model.eval()
        with torch.no_grad():
            points = data.get("points").to(self.device)
            inputs = data.get("inputs", torch.empty(points.size(0), 0)).to(self.device)
            points_iou = data.get("points_iou").to(self.device)
            df_iou = data.get("points_iou.df").to(self.device)
            inputs = add_key(inputs, data.get("inputs.ind"), "points", "index", device=self.device)
            points_iou = add_key(points_iou, data.get("points_iou.normalized"), "p", "p_n", device=self.device)

            kwargs = {}
            output = self.model(points_iou, inputs, **kwargs)
            output = torch.abs(output)

            df_iou_np = (df_iou <= self.threshold).cpu().numpy()
            df_iou_hat_np = (output <= self.threshold).cpu().numpy()
            iou = compute_iou(df_iou_np, df_iou_hat_np).mean()

        print(f"IOU[{iou:5.2f}]")
        self.tracker.track(
            iou,
            name="IOU",
            step=it,
            epoch=epoch_it,
            context={"subset": "train", "y0to1": True, "track": "current"},
        )
        return loss.item()

    def eval_step(self, data, epoch_it, it):
        """Performs an evaluation step.

        Args:
            data (dict): data dictionary
        """
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        # print(data)
        points = data.get("points").to(device)
        df = data.get("points.df").to(device)

        inputs = data.get("inputs", torch.empty(points.size(0), 0)).to(device)

        points_iou = data.get("points_iou").to(device)
        df_iou = data.get("points_iou.df").to(device)

        batch_size = points.size(0)

        kwargs = {}

        # add pre-computed index
        inputs = add_key(inputs, data.get("inputs.ind"), "points", "index", device=device)
        # add pre-computed normalized coordinates
        points = add_key(points, data.get("points.normalized"), "p", "p_n", device=device)
        points_iou = add_key(points_iou, data.get("points_iou.normalized"), "p", "p_n", device=device)

        # Compute iou
        print("VAL STEP")
        print({f"inputs.shape={inputs.shape}"})
        print({f"points_iou.shape={points_iou.shape}"})
        with torch.no_grad():
            output = self.model(points_iou, inputs, **kwargs)

        output = torch.abs(output)

        df_iou_np = (df_iou <= threshold).cpu().numpy()
        df_iou_hat_np = (output <= threshold).cpu().numpy()
        iou = compute_iou(df_iou_np, df_iou_hat_np).mean()
        self.tracker.track(
            iou,
            name="IOU",
            step=it,
            epoch=epoch_it,
            context={"subset": "val", "y0to1": True, "track": "current"},
        )
        eval_dict["iou"] = iou

        return eval_dict

    def compute_loss(self, data):
        """Computes the loss.

        Args:
            data (dict): data dictionary
        """
        device = self.device
        p = data.get("points").to(device)
        df = data.get("points.df").to(device)
        inputs = data.get("inputs", torch.empty(p.size(0), 0)).to(device)
        c = self.model.encode_inputs(inputs)
        kwargs = {}
        output = self.model.decode(p, c, **kwargs)
        output = torch.abs(output)
        loss = self.loss(output, df).sum(-1).mean()
        return output, loss

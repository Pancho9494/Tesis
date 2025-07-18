#  Ref: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
#  Ref: https://github.com/hansen7/OcCo/blob/master/OcCo_Torch/models/dgcnn_cls.py

import torch, torch.nn as nn, torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20, idx=None, extra_dim=False):
    batch_size, num_dims, num_points = x.size()
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if extra_dim is False:
            idx = knn(x, k=k)
        else:
            idx = knn(x[:, 6:], k=k)  # idx = knn(x[:, :3], k=k)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx += idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2 * num_dims, num_points, k)


class get_model(nn.Module):
    def __init__(self, args, num_channel=3, num_class=40, **kwargs):
        super(get_model, self).__init__()
        self.args = args
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

        self.conv1 = nn.Sequential(
            nn.Conv2d(num_channel * 2, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.linear1 = nn.Linear(args.emb_dims * 2, 512, bias=False)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, num_class)

    def forward(self, x):
        batch_size = x.size()[0]
        x = get_graph_feature(x, k=self.args.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.args.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.args.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.args.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class get_loss(torch.nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    @staticmethod
    def cal_loss(pred, gold, smoothing=True):
        """Calculate cross entropy loss, apply label smoothing if needed."""
        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size()[1]
            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)  # (num_points, num_class)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            loss = -(one_hot * log_prb).sum(dim=1).mean()  # ~ F.nll_loss(log_prb, gold)
        else:
            loss = F.cross_entropy(pred, gold, reduction="mean")

        return loss

    def forward(self, pred, target):
        return self.cal_loss(pred, target, smoothing=True)

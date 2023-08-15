"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import copy
from models.criterion import AngularPenaltySMLoss
# from criterion import AngularPenaltySMLoss
import sys

__all__ = ['resnet18']
# Sinkhorn Knopp
def sknopp(cZ, lamd=25, max_iters=100):
    with torch.no_grad():
        N_samples, N_centroids = cZ.shape # cZ is [N_samples, N_centroids]
        probs = F.softmax(cZ * lamd, dim=1).T # probs should be [N_centroids, N_samples]

        r = torch.ones((N_centroids, 1), device=probs.device) / N_centroids # desired row sum vector
        c = torch.ones((N_samples, 1), device=probs.device) / N_samples # desired col sum vector

        inv_N_centroids = 1. / N_centroids
        inv_N_samples = 1. / N_samples

        err = 1e3
        for it in range(max_iters):
            r = inv_N_centroids / (probs @ c)  # (N_centroids x N_samples) @ (N_samples, 1) = N_centroids x 1
            c_new = inv_N_samples / (r.T @ probs).T  # ((1, N_centroids) @ (N_centroids x N_samples)).t() = N_samples x 1
            if it % 10 == 0:
                err = torch.nansum(torch.abs(c / c_new - 1))
            c = c_new
            if (err < 1e-2):
                break

        # inplace calculations.
        probs *= c.squeeze()
        probs = probs.T # [N_samples, N_centroids]
        probs *= r.squeeze()

        return probs * N_samples # Soft assignments


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = NormedLinear(512 * block.expansion, num_classes)
        ##
        self.N_local = 32
        self.mem_projections = nn.Linear(1024, 512, bias=False) # para1: Memory size per client
        #self.centroids = NormedLinear(512 * block.expansion, num_classes) # global cluster centroids
        self.centroids = nn.Linear(512 * block.expansion, num_classes, bias=False)  # global cluster centroids
        self.local_centroids = nn.Linear(512 * block.expansion, self.N_local, bias=False)  # must be defined last
        # self.global_labeled_centroids = nn.Linear(512 * block.expansion, 10, bias=False)  # labeled data feature centroids
        self.local_labeled_centroids = nn.Linear(512 * block.expansion, num_classes, bias=False) # labeled data feature centroids
        self.T = 0.1
        self.labeled_num = 6

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    @torch.no_grad()
    def update_memory(self, F):
        N = F.shape[0]
        # Shift memory [D, m_size]
        self.mem_projections.weight.data[:, :-N] = self.mem_projections.weight.data[:, N:].detach().clone()
        # Transpose LHS [D, bsize]
        self.mem_projections.weight.data[:, -N:] = F.T.detach().clone()

    # Local clustering (happens at the client after every training round; clusters are made equally sized via Sinkhorn-Knopp, satisfying K-anonymity)
    def local_clustering(self, device=torch.device("cuda")):
        # Local centroids: [# of centroids, D]; local clustering input (mem_projections.T): [m_size, D]
        with torch.no_grad():
            Z = self.mem_projections.weight.data.T.detach().clone()
            centroids = Z[np.random.choice(Z.shape[0], self.N_local, replace=False)]
            local_iters = 5
            # clustering
            for it in range(local_iters):
                assigns = sknopp(Z @ centroids.T, max_iters=10)
                choice_cluster = torch.argmax(assigns, dim=1)
                for index in range(self.N_local):
                    selected = torch.nonzero(choice_cluster == index).squeeze()
                    selected = torch.index_select(Z, 0, selected)
                    if selected.shape[0] == 0:
                        selected = Z[torch.randint(len(Z), (1,))]
                    centroids[index] = F.normalize(selected.mean(dim=0), dim=0)

        # Save local centroids
        self.local_centroids.weight.data.copy_(centroids.to(device))

    # Global clustering (happens only on the server; see Genevay et al. for full details on the algorithm)
    # def global_clustering(self, Z1, nG=1., nL=1.):
    #     N = Z1.shape[0] # Z has dimensions [m_size * n_clients, D]
    #     # Optimizer setup
    #     optimizer = torch.optim.SGD(self.centroids.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    #     train_loss = 0.
    #     total_rounds = 500
    #     for round_idx in range(total_rounds):
    #         with torch.no_grad():
    #             # Cluster assignments from Sinkhorn Knopp
    #             SK_assigns = sknopp(self.centroids(Z1))
    #         # Zero grad
    #         optimizer.zero_grad()
    #         # Predicted cluster assignments [N, N_centroids] = local centroids [N, D] x global centroids [D, N_centroids]
    #         probs1 = F.softmax(self.centroids(F.normalize(Z1, dim=1)) / self.T, dim=1)
    #         # Match predicted assignments with SK assignments
    #         loss = - F.cosine_similarity(SK_assigns, probs1, dim=-1).mean()
    #         # Train
    #         loss.backward()
    #         optimizer.step()
    #         with torch.no_grad():
    #             #self.centroids.weight.copy_(self.centroids.weight.data.clone()) # Not Normalize centroids
    #             self.centroids.weight.copy_(F.normalize(self.centroids.weight.data.clone(), dim=1)) # Normalize centroids
    #             train_loss += loss.item()
    #     ######
    ###
    # Global clustering (happens only on the server; see Genevay et al. for full details on the algorithm)
    def global_clustering(self, Z1, nG=1., nL=1.):
        N = Z1.shape[0] # Z has dimensions [m_size * n_clients, D]
        # Optimizer setup
        optimizer = torch.optim.SGD(self.centroids.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        train_loss = 0.
        total_rounds = 500
        angular_criterion = AngularPenaltySMLoss()
        for round_idx in range(total_rounds):
            with torch.no_grad():
                # Cluster assignments from Sinkhorn Knopp
                SK_assigns = sknopp(self.centroids(Z1))
            # Zero grad
            optimizer.zero_grad()
            # Predicted cluster assignments [N, N_centroids] = local centroids [N, D] x global centroids [D, N_centroids]
            probs1 = F.softmax(self.centroids(F.normalize(Z1, dim=1)) / self.T, dim=1)
            ## 增加 Prototype距离 ##
            # cos_output = self.centroids(F.normalize(Z1, dim=1))
            # SK_target = np.argmax(SK_assigns.cpu().numpy(), axis=1)
            # angular_loss = angular_criterion(cos_output, SK_target)
            # print("angular_loss: ", angular_loss)
            ######################
            # Match predicted assignments with SK assignments
            cos_loss = F.cosine_similarity(SK_assigns, probs1, dim=-1).mean()
            loss = - cos_loss #+ angular_loss
            print("F.cosine_similarity: ", cos_loss)
            # Train
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                #self.centroids.weight.copy_(self.centroids.weight.data.clone()) # Not Normalize centroids
                self.centroids.weight.copy_(F.normalize(self.centroids.weight.data.clone(), dim=1)) # Normalize centroids
                train_loss += loss.item()
        #sys.exit(0)
        ######
    ###
    def set_labeled_feature_centroids(self, device=torch.device("cuda")):
        assignment = [999 for _ in range(self.labeled_num)]
        not_assign_list = [i for i in range(10)]
        # 让labeled feature中心点替换 最接近的 self.centroids参数 #
        C = self.centroids.weight.data.detach().clone() # [10,512]
        labeled_feature_centroids = self.local_labeled_centroids.weight.data[:self.labeled_num].detach().clone() # [labeled_num,512]
        copy_labeled_feature_centroids = copy.deepcopy(labeled_feature_centroids)
        copy_C = copy.deepcopy(C)
        #
        # C_norm = C / torch.norm(C, 2, 1, keepdim=True) #采用欧氏距离时，C此时没有归一化
        C_norm = C
        labeled_norm = labeled_feature_centroids / torch.norm(labeled_feature_centroids, 2, 1, keepdim=True)
        cosine_dist = torch.mm(labeled_norm, C_norm.t()) # [labeled_num, 10]
        vals, pos_idx = torch.topk(cosine_dist, 2, dim=1)
        pos_idx_1 = pos_idx[:, 0].cpu().numpy().flatten().tolist() # top1 [labeled_num]
        pos_idx_2 = pos_idx[:, 1].cpu().numpy().flatten().tolist() # top2 [labeled_num]
        print("cosine_dist: ", cosine_dist)
        print("pos_idx: ", pos_idx)
        print("pos_idx_1: ", pos_idx_1)
        print("pos_idx_2: ", pos_idx_2)
        #
        for idx in range(self.labeled_num):
            if pos_idx_1[idx] not in assignment:
                assignment[idx] = pos_idx_1[idx]
                not_assign_list.remove(assignment[idx])
                #C[assignment[idx]] = labeled_feature_centroids[idx]
            else:
                assignment[idx] = pos_idx_2[idx]
                not_assign_list.remove(assignment[idx])
                #C[assignment[idx]] = labeled_feature_centroids[idx]
        # set labeled centroids at first
        for idx in range(10):
            if idx < self.labeled_num:
                # C[idx] = copy_labeled_feature_centroids[idx]  ##### use avg label data feature centroids
                C[idx] = copy_C[assignment[idx]] ##### use cluster centroids #####
                # C[idx] = labeled_norm[idx] * torch.norm(copy_C[assignment[idx]], 2)  ##### 用label data中心 加上 归一化
                # C[idx] = copy_C[assignment[idx]] * torch.norm(copy_labeled_feature_centroids[idx] , 2)#采用欧氏距离时，global中心不用归一化
                # if idx == 0:
                #     avg_norm = torch.norm(copy_labeled_feature_centroids[idx] , 2)
                # else:
                #     avg_norm = avg_norm + torch.norm(copy_labeled_feature_centroids[idx] , 2)
            else:
                C[idx] = copy_C[not_assign_list[idx - self.labeled_num]]
                # C[idx] = copy_C[not_assign_list[idx - self.labeled_num]] * (avg_norm/self.labeled_num)#采用欧氏距离时，global中心不用归一化
        #
        self.centroids.weight.data.copy_(C.to(device))
        # self.centroids.weight.data.copy_(F.normalize(C.to(device), dim=1))
        # return -1
    ###

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out_linear = self.linear(out)
        #
        tZ1 = F.normalize(out, dim=1)
        # Update target memory
        with torch.no_grad():
            self.update_memory(tZ1) # New features are [bsize, D]
        #
        return out_linear, out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return 10 * out
import sys
import copy
import argparse
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
import open_world_cifar as datasets
import client_open_world_cifar as client_datasets
from utils import cluster_acc, AverageMeter, entropy, MarginLoss, accuracy, TransformTwice, cluster_acc_w
from sklearn import metrics
import numpy as np
import os
# from utils_cluster import
from torch.utils.tensorboard import SummaryWriter
from itertools import cycle

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def receive_models(clients_model):
    global uploaded_weights
    global uploaded_models
    uploaded_weights = []
    uploaded_models = []
    for model in clients_model:
        uploaded_weights.append( 1.0 / len(clients_model))
        # self.uploaded_models.append(copy.deepcopy(client.model.parameters()))
        uploaded_models.append(model.parameters())


def add_parameters(w, client_model):
    for (name, server_param), client_param in zip(global_model.named_parameters(), client_model):
        if "centroids" not in name:
            server_param.data += client_param.data.clone() * w
        if "local_labeled_centroids" in name:
            server_param.data += client_param.data.clone() * w
            # print("Averaged layer name: ", name)


def aggregate_parameters():
    for name, param in global_model.named_parameters():
        if "centroids" not in name:
            param.data = torch.zeros_like(param.data)
        if "local_labeled_centroids" in name:
            param.data = torch.zeros_like(param.data)
            # print("zeros_liked layer name: ", name)
    for w, client_model in zip(uploaded_weights, uploaded_models):
        add_parameters(w, client_model)


def train(args, model, device, train_label_loader, train_unlabel_loader, optimizer, m, epoch, tf_writer, client_id, global_round, last_unlabel_num):
    model.local_labeled_centroids.weight.data.zero_()  # model.local_labeled_centroids.weight.data: torch.Size([10, 512])
    labeled_samples_num = [0 for _ in range(10)]
    unlabel_samples_num = [0 for _ in range(10)]
    model.train()
    bce = nn.BCELoss()
    m = min(m, 0.5)
    # m = 0
    ce = MarginLoss(m=-1*m)
    unlabel_ce = MarginLoss(m=0) #(m=-1*m)
    unlabel_loader_iter = cycle(train_unlabel_loader)
    bce_losses = AverageMeter('bce_loss', ':.4e')
    ce_losses = AverageMeter('ce_loss', ':.4e')
    entropy_losses = AverageMeter('entropy_loss', ':.4e')
    #
    np_cluster_preds = np.array([]) # cluster_preds
    np_unlabel_targets = np.array([])
    #
    print("################### last_unlabel_num: ", last_unlabel_num)
    ## 各个类的不确定性权重（固定值）
    beta = 0.2
    Nk = last_unlabel_num
    Nmax = max(last_unlabel_num)
    if Nmax > 0:
        p_weight = [beta ** (1 - Nk[i] / Nmax) for i in range(10)]
    else:
        p_weight = [beta for i in range(10)]
    print("################### p_weight: ", p_weight)
    #
    for batch_idx, ((x, x2), target) in enumerate(train_label_loader):
        ((ux, ux2), unlabel_target) = next(unlabel_loader_iter)

        x = torch.cat([x, ux], 0)
        x2 = torch.cat([x2, ux2], 0)
        #
        labeled_len = len(target)
        # print("labeled_len: ", labeled_len)

        x, x2, target = x.to(device), x2.to(device), target.to(device)
        optimizer.zero_grad()
        output, feat = model(x) #output: [batch size, 10]; feat: [batch size, 512]
        output2, feat2 = model(x2)
        prob = F.softmax(output, dim=1)
        reg_prob = F.softmax(output[labeled_len:], dim=1) # unlabel data's prob
        prob2 = F.softmax(output2, dim=1)
        reg_prob2 = F.softmax(output2[labeled_len:], dim=1)  # unlabel data's prob

        # update local_labeled_centroids
        for feature, true_label in zip(feat[:labeled_len].detach().clone(), target):
            labeled_samples_num[true_label] += 1
            model.local_labeled_centroids.weight.data[true_label] += feature
        # print("before model.local_labeled_centroids.weight.data: ", model.local_labeled_centroids.weight.data)
        # for idx, (feature_centroid, num) in enumerate(zip(model.local_labeled_centroids.weight.data, labeled_samples_num)): #########################
        #     if num > 0:                                                                                                       #########################
        #         model.local_labeled_centroids.weight.data[idx] = feature_centroid/num                                        #########################
        # print("model.local_labeled_centroids.weight.data size: ", model.local_labeled_centroids.weight.data.size())
        # print("model.local_labeled_centroids.weight.data: ", model.local_labeled_centroids.weight.data)
        # print("labeled_samples_num: ", labeled_samples_num)
        # L_reg:  reg_prob中每一行的预测label
        copy_reg_prob1 = copy.deepcopy(reg_prob.detach())
        copy_reg_prob2 = copy.deepcopy(reg_prob2.detach())
        reg_label1 = np.argmax(copy_reg_prob1.cpu().numpy(), axis=1)
        reg_label2 = np.argmax(copy_reg_prob2.cpu().numpy(), axis=1)
        ### 制作target, target 除了 label=1 外与 reg_prob 一致
        for idx, (label, oprob) in enumerate(zip(reg_label1, copy_reg_prob1)):
            copy_reg_prob1[idx][label] = 1
            unlabel_samples_num[label] += 1
        for idx, (label, oprob) in enumerate(zip(reg_label2, copy_reg_prob2)):
            copy_reg_prob2[idx][label] = 1
        #
        L1_loss = nn.L1Loss()
        L_reg1 = 0.0
        L_reg2 = 0.0
        for idx, (ooutput, otarget, label) in enumerate(zip(reg_prob, copy_reg_prob1, reg_label1)):
            L_reg1 = L_reg1 + L1_loss(reg_prob[idx], copy_reg_prob1[idx]) * p_weight[label]
        for idx, (ooutput, otarget, label) in enumerate(zip(reg_prob2, copy_reg_prob2, reg_label2)):
            L_reg2 = L_reg2 + L1_loss(reg_prob2[idx], copy_reg_prob2[idx]) * p_weight[label]
        L_reg1 = L_reg1 / len(reg_label1)
        L_reg2 = L_reg2 / len(reg_label2)
        #### Ours loss end
        ## 欧氏距离 ########################################################################
        # C = model.centroids.weight.data.detach().clone()
        # Z1 = feat.detach()
        # Z2 = feat2.detach()
        # cP1 = euclidean_dist(Z1, C)
        # cZ2 = euclidean_dist(Z2, C)
        ## Cluster loss begin (Orchestra) # cos-similarity ###############################
        C = model.centroids.weight.data.detach().clone().T
        Z1 = F.normalize(feat, dim=1)
        Z2 = F.normalize(feat2, dim=1)
        cP1 = Z1 @ C
        cZ2 = Z2 @ C
        ##
        tP1 = F.softmax(cP1 / model.T, dim=1)
        confidence_cluster_pred, cluster_pred = tP1.max(1) # cluster_pred: [512]; target: [170]
        tP2 = F.softmax(cZ2 / model.T, dim=1)
        #logpZ2 = torch.log(F.softmax(cZ2 / model.T, dim=1))
        # Clustering loss
        #L_cluster = - torch.sum(tP1 * logpZ2, dim=1).mean()
        # print("L_cluster: ", L_cluster)
        ## Cluster loss end (Orchestra)
        ### 统计 cluster_pred (伪标签，cluster id) 置信度 ###
        confidence_list = [0 for _ in range(10)]
        num_of_cluster = [0 for _ in range(10)]
        #mask_tmp = np.array([])
        for confidence, cluster_id in zip(confidence_cluster_pred[labeled_len:], cluster_pred[labeled_len:]):
            confidence_list[cluster_id] = confidence_list[cluster_id] + confidence
            num_of_cluster[cluster_id] = num_of_cluster[cluster_id] + 1
        for cluster_id, (sum_confidence, num) in enumerate(zip(confidence_list, num_of_cluster)):
            if num > 0:
                confidence_list[cluster_id] = confidence_list[cluster_id].cpu().detach().numpy()/num
                confidence_list[cluster_id] = np.around(confidence_list[cluster_id], 4) #保留小数点后4位
        #mask_tmp = np.append(mask_tmp, confidence_cluster_pred[labeled_len:].cpu().detach().numpy())
        threshold = 0.95
        # confidence_mask = mask_tmp > threshold
        confidence_mask = (confidence_cluster_pred[labeled_len:] > threshold)
        confidence_mask = torch.nonzero(confidence_mask)
        confidence_mask = torch.squeeze(confidence_mask)
        if client_id == 0:
            print("confidence_mask: ", confidence_mask)
        #print("confidence_mask: ", confidence_mask)
        #sys.exit(0)
        #if (args.epochs * global_round + epoch) % 5 == 0:
        print("global round: ", global_round, ";   client_id: ", client_id, ";   confidence_list: ", confidence_list)

        # calculate distance
        feat_detach = feat.detach()
        feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
        cosine_dist = torch.mm(feat_norm, feat_norm.t())
        labeled_len = len(target)

        pos_pairs = []
        target_np = target.cpu().numpy()
        
        # label part
        for i in range(labeled_len):
            target_i = target_np[i]
            idxs = np.where(target_np == target_i)[0]
            if len(idxs) == 1:
                pos_pairs.append(idxs[0])
            else:
                selec_idx = np.random.choice(idxs, 1)
                while selec_idx == i:
                    selec_idx = np.random.choice(idxs, 1)
                pos_pairs.append(int(selec_idx))

        # unlabel part
        unlabel_cosine_dist = cosine_dist[labeled_len:, :]
        vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
        # print(pos_idx.size())
        # print(pos_idx)
        pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
        pos_pairs.extend(pos_idx) #pos_pairs size: [512,1]

        # bce + L_cluster
        cluster_pos_prob = tP2[pos_pairs, :] #cluster_pos_prob size: [512,10]
        # bce
        # cluster_pos_sim = torch.bmm(tP1.view(args.batch_size, 1, -1), cluster_pos_prob.view(args.batch_size, -1, 1)).squeeze()
        # cluster_ones = torch.ones_like(cluster_pos_sim)
        # cluster_bce_loss = bce(cluster_pos_sim, cluster_ones)
        # cross-entropy
        logcluster_pos_prob = torch.log(cluster_pos_prob)
        L_cluster = - torch.sum(tP1 * logcluster_pos_prob, dim=1).mean() #[170(label)/512-170(unlabel)]
        #
        pos_prob = prob2[pos_pairs, :]
        pos_sim = torch.bmm(prob.view(args.batch_size, 1, -1), pos_prob.view(args.batch_size, -1, 1)).squeeze()
        ones = torch.ones_like(pos_sim)
        bce_loss = bce(pos_sim, ones)
        ce_loss = ce(output[:labeled_len], target)
        # unlabel ce loss
        # unlabel_ce_loss = unlabel_ce(output[labeled_len:], cluster_pred[labeled_len:])
        ###
        #print("1:",output[labeled_len:].index_select(0,confidence_mask).size())
        #print("2:",cluster_pred[labeled_len:].index_select(0,confidence_mask).size())
        ####
        unlabel_ce_loss = unlabel_ce(output[labeled_len:].index_select(0,confidence_mask) , cluster_pred[labeled_len:].index_select(0,confidence_mask))
        np_cluster_preds = np.append(np_cluster_preds, cluster_pred[labeled_len:].cpu().numpy())
        np_unlabel_targets = np.append(np_unlabel_targets, unlabel_target.cpu().numpy())
        #
        entropy_loss = entropy(torch.mean(prob, 0))
        
        #loss = - entropy_loss + ce_loss + bce_loss
        # loss = ce_loss
        if global_round > 4: #4
            if global_round > 6: #6
                loss = - entropy_loss + ce_loss + bce_loss + 0.5 * L_cluster + unlabel_ce_loss #- 2 * L_reg1 ####+ 2 * L_reg2  # + L_cluster # 调整L_reg倍率
            else:
                loss = - entropy_loss + ce_loss + bce_loss + 0.5 * L_cluster #+ 2 * L_reg1 + 2 * L_reg2 #+ L_cluster # 调整L_reg倍率
        else:
            loss = - entropy_loss + ce_loss + bce_loss #+ 2 * L_reg1 + 2 * L_reg2 # 调整L_reg倍率
        if client_id == 0:
            print("entropy_loss: ", entropy_loss)
            print("ce_loss: ", ce_loss)
            print("bce_loss: ", bce_loss)
            print("L_cluster: ", L_cluster)
            print("unlabel_ce_loss: ", unlabel_ce_loss)
            print("L_reg1: ", L_reg1)
        # print("L_reg2: ", 2 * L_reg2)
        # sys.exit(0)

        bce_losses.update(bce_loss.item(), args.batch_size)
        ce_losses.update(ce_loss.item(), args.batch_size)
        entropy_losses.update(entropy_loss.item(), args.batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #if ((args.epochs * global_round + epoch) % 10 == 0) and (client_id == 0):
    for idx, (feature_centroid, num) in enumerate(zip(model.local_labeled_centroids.weight.data, labeled_samples_num)):
        if num > 0:
            model.local_labeled_centroids.weight.data[idx] = feature_centroid/num
    #if client_id == 0:
        #unlabel_acc, w_unlabel_acc = cluster_acc_w(np.array(cluster_pred[labeled_len:].cpu().numpy()), np.array(unlabel_target.cpu().numpy()))
    np_cluster_preds = np_cluster_preds.astype(int)
    unlabel_acc, w_unlabel_acc = cluster_acc_w(np_cluster_preds, np_unlabel_targets)
    print("unlabel_acc: ", unlabel_acc)
    print("w_unlabel_acc: ", w_unlabel_acc)
        #print("unlabel target: ", unlabel_target)
        #print("unlabel cluster_pred: ", cluster_pred[labeled_len:])
    #sys.exit(0)

    # tf_writer.add_scalar('client{}/loss/bce'.format(client_id), bce_losses.avg, args.epochs * global_round + epoch)
    # tf_writer.add_scalar('client{}/loss/ce'.format(client_id), ce_losses.avg, args.epochs * global_round + epoch)
    # tf_writer.add_scalar('client{}/loss/entropy'.format(client_id), entropy_losses.avg, args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/loss'.format(client_id), {"bce": bce_losses.avg}, args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/loss'.format(client_id), {"ce": ce_losses.avg}, args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/loss'.format(client_id), {"entropy": entropy_losses.avg}, args.epochs * global_round + epoch)
    print("################### unlabel_samples_num: ", unlabel_samples_num)
    return unlabel_samples_num


def test(args, model, labeled_num, device, test_loader, epoch, tf_writer, client_id, global_round):
    model.eval()
    preds = np.array([])
    cluster_preds = np.array([]) # cluster_preds
    targets = np.array([])
    confs = np.array([])
    with torch.no_grad():
        C = model.centroids.weight.data.detach().clone().T
        for batch_idx, (x, label) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            output, feat = model(x)
            prob = F.softmax(output, dim=1)
            conf, pred = prob.max(1)
            # cluster pred
            Z1 = F.normalize(feat, dim=1)
            cP1 = Z1 @ C
            tP1 = F.softmax(cP1 / model.T, dim=1)
            _, cluster_pred = tP1.max(1) # return #1: max data    #2: max data index
            #
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())
            cluster_preds = np.append(cluster_preds, cluster_pred.cpu().numpy())
            confs = np.append(confs, conf.cpu().numpy())
    targets = targets.astype(int)
    preds = preds.astype(int)
    cluster_preds = cluster_preds.astype(int)

    seen_mask = targets < labeled_num
    unseen_mask = ~seen_mask
    ## preds <-> cluster_preds ##
    origin_preds = preds
    # preds = cluster_preds
    ## local_unseen_mask (4) ##
    local_unseen_mask_4 = targets == 4
    local_unseen_acc_4 = cluster_acc(preds[local_unseen_mask_4], targets[local_unseen_mask_4])
    ## local_unseen_mask (4) ##
    local_unseen_mask_5 = targets == 5
    local_unseen_acc_5 = cluster_acc(preds[local_unseen_mask_5], targets[local_unseen_mask_5])
    ## local_unseen_mask (4) ##
    local_unseen_mask_6 = targets == 6
    local_unseen_acc_6 = cluster_acc(preds[local_unseen_mask_6], targets[local_unseen_mask_6])
    ## local_unseen_mask (4) ##
    local_unseen_mask_7 = targets == 7
    local_unseen_acc_7 = cluster_acc(preds[local_unseen_mask_7], targets[local_unseen_mask_7])
    ## local_unseen_mask (4) ##
    local_unseen_mask_8 = targets == 8
    local_unseen_acc_8 = cluster_acc(preds[local_unseen_mask_8], targets[local_unseen_mask_8])
    ## local_unseen_mask (4) ##
    local_unseen_mask_9 = targets == 9
    local_unseen_acc_9 = cluster_acc(preds[local_unseen_mask_9], targets[local_unseen_mask_9])
    ## global_unseen_mask (5-9) ##
    global_unseen_mask = targets > labeled_num
    global_unseen_acc = cluster_acc(preds[global_unseen_mask], targets[global_unseen_mask])
    ##
    # overall_acc = cluster_acc(preds, targets)
    overall_acc, w_overall_acc = cluster_acc_w(origin_preds, targets)
    if ((args.epochs * global_round + epoch) % 10 == 0) and (client_id == 0):
        print("w_overall_acc: ", w_overall_acc)
    # cluster_acc
    overall_cluster_acc = cluster_acc(cluster_preds, targets)
    #
    # seen_acc = accuracy(preds[seen_mask], targets[seen_mask])
    seen_acc = accuracy(origin_preds[seen_mask], targets[seen_mask])
    #
    unseen_acc, w_unseen_acc = cluster_acc_w(preds[unseen_mask], targets[unseen_mask])
    if ((args.epochs * global_round + epoch) % 10 == 0) and (client_id == 0):
        print("w_unseen_acc: ", w_unseen_acc)
    unseen_nmi = metrics.normalized_mutual_info_score(targets[unseen_mask], preds[unseen_mask])
    mean_uncert = 1 - np.mean(confs)
    print('epoch {}, Client id {}, Test overall acc {:.4f}, Test overall cluster acc {:.4f}, seen acc {:.4f}, unseen acc {:.4f}, local_unseen acc {:.4f}, global_unseen acc {:.4f}'.format(epoch, client_id, overall_acc, overall_cluster_acc, seen_acc, unseen_acc, local_unseen_acc_6, global_unseen_acc))
    # tf_writer.add_scalar('client{}/acc/overall'.format(client_id), overall_acc, args.epochs * global_round + epoch)
    # tf_writer.add_scalar('client{}/acc/seen'.format(client_id), seen_acc, args.epochs * global_round + epoch)
    # tf_writer.add_scalar('client{}/acc/unseen'.format(client_id), unseen_acc, args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"overall": overall_acc}, args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"seen": seen_acc}, args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"unseen": unseen_acc}, args.epochs * global_round + epoch)
    ##
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"local_unseen_4": local_unseen_acc_4}, args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"local_unseen_5": local_unseen_acc_5}, args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"local_unseen_6": local_unseen_acc_6}, args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"local_unseen_7": local_unseen_acc_7}, args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"local_unseen_8": local_unseen_acc_8}, args.epochs * global_round + epoch)
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"local_unseen_9": local_unseen_acc_9}, args.epochs * global_round + epoch)
    # #
    tf_writer.add_scalars('client{}/acc'.format(client_id), {"global_unseen": global_unseen_acc}, args.epochs * global_round + epoch)
    ##
    tf_writer.add_scalar('client{}/nmi/unseen'.format(client_id), unseen_nmi, args.epochs * global_round + epoch)
    tf_writer.add_scalar('client{}/uncert/test'.format(client_id), mean_uncert, args.epochs * global_round + epoch)
    return mean_uncert


def main():
    parser = argparse.ArgumentParser(description='orca')
    parser.add_argument('--milestones', nargs='+', type=int, default=[140, 180])
    parser.add_argument('--dataset', default='cifar100', help='dataset setting')
    parser.add_argument('--clients-num', default=5, type=int)
    parser.add_argument('--global-rounds', default=20, type=int)
    parser.add_argument('--labeled-num', default=50, type=int)
    parser.add_argument('--labeled-ratio', default=0.5, type=float)
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--exp_root', type=str, default='./results/')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    print("-------------------- use ",device,"-----------")
    args.savedir = os.path.join(args.exp_root, args.name)
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    if args.dataset == 'cifar10':
        #train_label_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']), exist_label_list=[0,1,2,3,4,5,6,7,8,9], clients_num=args.clients_num)
        train_label_set = datasets.OPENWORLDCIFAR10(root='./datasets', labeled=True, labeled_num=10, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']))
        #print("self.data: ", train_label_set.data.shape)
        #sys.exit(0)
        # train_unlabel_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']), unlabeled_idxs=train_label_set.unlabeled_idxs, exist_label_list=[0,1,2,3,4,5], clients_num=args.clients_num)
        # test_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=datasets.dict_transform['cifar_test'], unlabeled_idxs=train_label_set.unlabeled_idxs, exist_label_list=[0,1,2,3,4,5], clients_num=args.clients_num)
        num_classes = 10
        ### prepare clients dataset ###
        ## 子集
        # exist_label_list=[[0,1,2,3,4,5], [0,1,2,3,4,6], [0,1,2,3,4,7], [0,1,2,3,4,8], [0,1,2,3,4,9]]
        # clients_labeled_num = [4, 4, 4, 4, 4]
        ## 全集
        exist_label_list=[[0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,8], [0,1,2,3,4,5,6,8], [0,1,2,3,4,5,6,9]]
        clients_labeled_num = [6, 6, 6, 6, 6]
        ##
        ## 80%
        # exist_label_list=[[0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9]]
        # clients_labeled_num = [8, 8, 8, 8, 8]
        ##
        clients_train_label_set = []
        clients_train_unlabel_set = []
        clients_test_set = []
        for i in range(args.clients_num):
            client_train_label_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=True, labeled_num=clients_labeled_num[i],
                                                        labeled_ratio=args.labeled_ratio, download=True,
                                                        transform=TransformTwice(
                                                            datasets.dict_transform['cifar_train']), exist_label_list=exist_label_list[i], clients_num=args.clients_num)
            client_train_unlabel_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False,
                                                          labeled_num=clients_labeled_num[i],
                                                          labeled_ratio=args.labeled_ratio, download=True,
                                                          transform=TransformTwice(
                                                              datasets.dict_transform['cifar_train']),
                                                          unlabeled_idxs=client_train_label_set.unlabeled_idxs, exist_label_list=exist_label_list[i], clients_num=args.clients_num)
            # client_test_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num,
            #                                             labeled_ratio=args.labeled_ratio, download=True,
            #                                             transform=datasets.dict_transform['cifar_test'],
            #                                             unlabeled_idxs=client_train_label_set.unlabeled_idxs,
            #                                             exist_label_list=exist_label_list[i],
            #                                             clients_num=args.clients_num)
            client_test_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                                                        labeled_ratio=args.labeled_ratio, download=True,
                                                        transform=datasets.dict_transform['cifar_test'],
                                                        unlabeled_idxs=train_label_set.unlabeled_idxs,
                                                        exist_label_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                        clients_num=args.clients_num)

            clients_train_label_set.append(client_train_label_set)
            clients_train_unlabel_set.append(client_train_unlabel_set)
            clients_test_set.append(client_test_set)
        ###
    elif args.dataset == 'cifar100':
        train_label_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']))
        train_unlabel_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']), unlabeled_idxs=train_label_set.unlabeled_idxs)
        test_set = datasets.OPENWORLDCIFAR100(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=datasets.dict_transform['cifar_test'], unlabeled_idxs=train_label_set.unlabeled_idxs)
        num_classes = 100
    elif args.dataset == 'cinic10':
        #train_label_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=True, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']), exist_label_list=[0,1,2,3,4,5,6,7,8,9], clients_num=args.clients_num)
        train_label_set = datasets.OPENWORLDCINIC10(root='./datasets', labeled=True, labeled_num=10, labeled_ratio=0.8, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']))
        #print("self.data: ", train_label_set.data.shape)
        #sys.exit(0)
        # train_unlabel_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=TransformTwice(datasets.dict_transform['cifar_train']), unlabeled_idxs=train_label_set.unlabeled_idxs, exist_label_list=[0,1,2,3,4,5], clients_num=args.clients_num)
        # test_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num, labeled_ratio=args.labeled_ratio, download=True, transform=datasets.dict_transform['cifar_test'], unlabeled_idxs=train_label_set.unlabeled_idxs, exist_label_list=[0,1,2,3,4,5], clients_num=args.clients_num)
        num_classes = 10
        ### prepare clients dataset ###
        ## 子集
        # exist_label_list=[[0,1,2,3,4,5], [0,1,2,3,4,6], [0,1,2,3,4,7], [0,1,2,3,4,8], [0,1,2,3,4,9]]
        # clients_labeled_num = [4, 4, 4, 4, 4]
        ## 全集
        exist_label_list=[[0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,8], [0,1,2,3,4,5,6,8], [0,1,2,3,4,5,6,9]]
        clients_labeled_num = [6, 6, 6, 6, 6]
        ##
        clients_train_label_set = []
        clients_train_unlabel_set = []
        clients_test_set = []
        for i in range(args.clients_num):
            client_train_label_set = client_datasets.OPENWORLDCINIC10(root='./datasets', labeled=True, labeled_num=clients_labeled_num[i],
                                                        labeled_ratio=args.labeled_ratio, download=True,
                                                        transform=TransformTwice(
                                                            datasets.dict_transform['cifar_train']), exist_label_list=exist_label_list[i], clients_num=args.clients_num)
            client_train_unlabel_set = client_datasets.OPENWORLDCINIC10(root='./datasets', labeled=False,
                                                          labeled_num=clients_labeled_num[i],
                                                          labeled_ratio=args.labeled_ratio, download=True,
                                                          transform=TransformTwice(
                                                              datasets.dict_transform['cifar_train']),
                                                          unlabeled_idxs=client_train_label_set.unlabeled_idxs, exist_label_list=exist_label_list[i], clients_num=args.clients_num)
            # client_test_set = client_datasets.OPENWORLDCIFAR10(root='./datasets', labeled=False, labeled_num=args.labeled_num,
            #                                             labeled_ratio=args.labeled_ratio, download=True,
            #                                             transform=datasets.dict_transform['cifar_test'],
            #                                             unlabeled_idxs=client_train_label_set.unlabeled_idxs,
            #                                             exist_label_list=exist_label_list[i],
            #                                             clients_num=args.clients_num)
            client_test_set = client_datasets.OPENWORLDCINIC10(root='./datasets', labeled=False, labeled_num=args.labeled_num,
                                                        labeled_ratio=args.labeled_ratio, download=True,
                                                        transform=datasets.dict_transform['cifar_test'],
                                                        unlabeled_idxs=train_label_set.unlabeled_idxs,
                                                        exist_label_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                                        clients_num=args.clients_num)

            clients_train_label_set.append(client_train_label_set)
            clients_train_unlabel_set.append(client_train_unlabel_set)
            clients_test_set.append(client_test_set)
        ###
    else:
        warnings.warn('Dataset is not listed')
        return

    #
    clients_labeled_batch_size = []
    for i in range(args.clients_num):
        labeled_len = len(clients_train_label_set[i])
        unlabeled_len = len(clients_train_unlabel_set[i])
        labeled_batch_size = int(args.batch_size * labeled_len / (labeled_len + unlabeled_len))
        clients_labeled_batch_size.append(labeled_batch_size)

    # Initialize the splits  # train_label_loader->client_train_label_loader[];   train_unlabel_loader->client_train_unlabel_loader[]
    client_train_label_loader = []
    client_train_unlabel_loader = []
    client_test_loader = []
    for i in range(args.clients_num): # train_label_loader->client_train_label_loader[];   train_unlabel_loader->client_train_unlabel_loader[]
        train_label_loader = torch.utils.data.DataLoader(clients_train_label_set[i], batch_size=clients_labeled_batch_size[i], shuffle=True, num_workers=2, drop_last=True)
        train_unlabel_loader = torch.utils.data.DataLoader(clients_train_unlabel_set[i], batch_size=args.batch_size - clients_labeled_batch_size[i], shuffle=True, num_workers=2, drop_last=True)
        client_train_label_loader.append(train_label_loader)
        client_train_unlabel_loader.append(train_unlabel_loader)

        test_loader = torch.utils.data.DataLoader(clients_test_set[i], batch_size=100, shuffle=False, num_workers=1)
        client_test_loader.append(test_loader)

    # Initialize the global_model ##############
    global global_model
    global_model = models.resnet18(num_classes=num_classes)
    global_model = global_model.to(device)
    if args.dataset == 'cifar10':
        state_dict = torch.load('./pretrained/simclr_cifar_10.pth.tar')
    elif args.dataset == 'cifar100':
        state_dict = torch.load('./pretrained/simclr_cifar_100.pth.tar')
    elif args.dataset == 'cinic10':
        state_dict = torch.load('./pretrained/simclr_cifar_10.pth.tar')
    global_model.load_state_dict(state_dict, strict=False)
    global_model = global_model.to(device)
    # Freeze the earlier filters
    for name, param in global_model.named_parameters():
        if 'linear' not in name and 'layer4' not in name:
            param.requires_grad = False
        if "centroids" in name:
            param.requires_grad = True
    ###########################################

    clients_model = [] # model->clients_model[client_id]
    clients_optimizer = [] # optimizer->clients_optimizer[client_id]
    clients_scheduler = [] # scheduler->clients_scheduler[client_id]
    clients_tf_writer = [] # tf_writer->clients_tf_writer[client_id]
    clients_unlabel_num = [] ########################################
    for i in range(args.clients_num):
        tmp_list = [0 for _ in range(10)]
        clients_unlabel_num.append(tmp_list)
        # First network intialization: pretrain the RotNet network
        # model = models.resnet18(num_classes=num_classes)
        # model = model.to(device)
        # if args.dataset == 'cifar10':
        #     state_dict = torch.load('./pretrained/simclr_cifar_10.pth.tar')
        # elif args.dataset == 'cifar100':
        #     state_dict = torch.load('./pretrained/simclr_cifar_100.pth.tar')
        # model.load_state_dict(state_dict, strict=False)
        model = copy.deepcopy(global_model)
        model = model.to(device)

        # Freeze the earlier filters
        for name, param in model.named_parameters():
            if 'linear' not in name and 'layer4' not in name:
                param.requires_grad = False
            if "centroids" in name:
                param.requires_grad = False
        clients_model.append(model)

        # Set the optimizer
        optimizer = optim.SGD(model.parameters(), lr=1e-1, weight_decay=5e-4, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
        clients_optimizer.append(optimizer)
        clients_scheduler.append(scheduler)

        #tf_writer = SummaryWriter(log_dir=args.savedir)
        #clients_tf_writer.append(tf_writer)
    tf_writer = SummaryWriter(log_dir=args.savedir)

    ## Start FedAvg training ##
    for global_round in range(args.global_rounds):
        print(" Start global_round {}: ".format(global_round))
        for client_id in range(args.clients_num):
            for epoch in range(args.epochs):
                mean_uncert = test(args, clients_model[client_id], args.labeled_num, device, client_test_loader[client_id], epoch, tf_writer, client_id, global_round)
                clients_unlabel_num[client_id] = train(args, clients_model[client_id], device, client_train_label_loader[client_id], client_train_unlabel_loader[client_id], clients_optimizer[client_id], mean_uncert, epoch, tf_writer, client_id, global_round, clients_unlabel_num[client_id])
                clients_scheduler[client_id].step()
            # local_clustering #
            clients_model[client_id].local_clustering(device=device)

        # receive_models
        receive_models(clients_model)

        # aggregate_parameters #global model 平均了所有client的模型参数
        aggregate_parameters()

        # Run global clustering
        for client_id in range(args.clients_num):
            for c_name, old_param in clients_model[client_id].named_parameters():
                if "local_centroids" in c_name:
                    if client_id == 0:
                        Z1 = np.array(copy.deepcopy(old_param.data.cpu().clone()))
                    else:
                        Z1 = np.concatenate((Z1, np.array(copy.deepcopy(old_param.data.cpu().clone()))), axis=0)
        Z1 = torch.tensor(Z1, device=device).T
        global_model.global_clustering(Z1.to(device).T) # update self.centroids in global model
        # set labeled data feature instead of self.centroids
        global_model.set_labeled_feature_centroids(device=device)
        #

        # download global model param
        # name_filters = ['linear', "mem_projections", "centroids", "local_centroids"]
        # name_filters = ['linear', "mem_projections", "local_centroids", "local_labeled_centroids"] #do not AVG FedRep
        name_filters = ["mem_projections", "local_centroids", "local_labeled_centroids"]  # do not AVG FedAVG
        for client_id in range(args.clients_num):
            for (g_name, new_param), (c_name, old_param) in zip(global_model.named_parameters(), clients_model[client_id].named_parameters()):
                if all(keyword not in g_name for keyword in name_filters):
                    old_param.data = new_param.data.clone()
                    # print("Download layer name: ", g_name)
            # sys.exit(0)

    ## finish train ##
    torch.save(global_model.state_dict(), './fedrep-trained-model/global.pth')
    for client_id in range(args.clients_num):
        torch.save(clients_model[client_id].state_dict(), './fedrep-trained-model/client{}-model.pth'.format(client_id))
    ## save model

if __name__ == '__main__':
    main()

# coding=utf-8
import torch
import random
import torch.nn as nn
from networks.MMAL5 import *
from utils.hash_loss import Hash_Loss
from utils.cal_map import *
from utils.triplet_loss import TripleLoss,construct_triplets
from utils.tools import *
from utils.dataset import *
from scipy.linalg import hadamard
import time
import torch.nn.functional as Fc
import itertools
from utils.read_data import Read_Dataset
from loguru import logger
import os

def get_config():
    config = {
        "info": "[dog-final(0.8)]",
        "resize_size": 448,
        "batch_size": 32,
        "dataset": "DOG",
        "epoch": 150,
        "test_map": 30,
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "bit_list": [32,64],
        "num_classes": 120,
        "N_list": [3, 2, 1],
        "num_workers": 8,
        "channels": 512,
        "init_lr": 0.001,
        "weight_decay": 1e-4,
        "num_parts": 3,
        "m": [0.7],
    }
    config = config_dataset(config)
    return config

# /*******    新增的损失     ******/
class CSQLoss(torch.nn.Module):
    def __init__(self, n_class,device, bit):
        super(CSQLoss, self).__init__()
        # self.is_single_label = config["dataset"] not in {"nuswide", "mirflickr", "coco"}
        self.hash_targets = self.get_hash_targets(n_class, bit).to(device)
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(device)
        self.criterion = torch.nn.BCELoss().to(device)

    def forward(self, u, y, ind):
        u = u.tanh()
        hash_center = self.label2center(y)
        center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))

        Q_loss = (u.abs() - 1).pow(2).mean()
        return center_loss + 0.0001 * Q_loss

    def label2center(self, y):
        hash_center = self.hash_targets[y.argmax(axis=1)]
        return hash_center

    # use algorithm 1 to generate hash centers
    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                # to find average/min  pairwise distance
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                # choose min(c) in the range of K/4 to K/3
                # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
                # but it is hard when bit is  small
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets
# /*******    新增的损失     ******/

def main(config, bit,margin):
    proposalN = sum(config["N_list"])
    device = config["device"]
    train_data = Stanford_Dogs(config["dataroot"], is_train=True)
    train_img_label = train_data.train_img_label
    num_train = len(train_img_label)
    train_label = torch.zeros(num_train)
    for i in range(num_train):
        cur_label = train_img_label[i][1]
        train_label[i] = torch.tensor(cur_label).float()
    train_dataloader, num_train, test_dataloader, num_test = Read_Dataset(config, num_train)

    n_select_train=num_train

    def seed_torch(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    seed = random.randint(1, 1000)
    seed_torch(seed)
    print(f'seed:{seed}')
    model = MainNet(proposalN= proposalN,num_classes=config["n_class"], channels=config["channels"], bit=bit)

    state_dict = torch.load('checkpoint/DOG_model_ft_190.t', map_location=device)['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    criterion = nn.CrossEntropyLoss()
    criterion3 = CSQLoss(n_class=config["num_classes"],device = config["device"],bit = bit)
    Best_mAP = 0
    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=config["init_lr"], momentum=0.9, weight_decay=config["weight_decay"])
    model = model.to(device)  ## 部署在GPU  todo

    # 开始训练
    B = torch.zeros(n_select_train, bit).to(device)
    B_label = torch.ones(n_select_train, config["n_class"]).to(device)

    C = torch.sign(torch.rand(config["n_class"], bit) - 0.5).to(device)
    center_label = torch.tensor(np.arange(0, config["n_class"], 1)).float()
    center_label_oh = one_hot_label(center_label, config["n_class"]).to(device)
    log = open("d:/lixue/fine-grained/code/MMAL+sRLH/logdog.txt", mode = "a+", encoding = "utf-8")
    for epoch in range(config["epoch"]):
        perm_index = np.random.permutation(num_train)
        select_samples_index = perm_index[0:n_select_train]
        cur_train_img_label = list(np.array(train_img_label)[select_samples_index])
        cur_dataloader = read_dataset(cur_train_img_label, config["resize_size"],
                                      config["batch_size"], config["dataset"], config["num_workers"], is_train=True,
                                      is_shuffle=True)
        cur_train_label = torch.zeros(n_select_train)
        for i in range(n_select_train):
            cur_label = cur_train_img_label[i][1]
            cur_train_label[i] = torch.tensor(int(cur_label))
        cur_train_label_oh = one_hot_label(cur_train_label, config["n_class"]).to(device)
        S = cur_train_label_oh.matmul(center_label_oh.t())
        S[S > 0] = 1
        S[S < 1] = -1
        r = S.sum() / (1 - S).sum()
        S = S * (1 + r) - r
        #S[S < 0] = -0.05
        loss = 0.0

        if epoch < 200:
            lr = config["init_lr"]
        elif epoch < 400:
            lr = config["init_lr"] * 0.1
        elif epoch < 600:
            lr = config["init_lr"] * 0.01
        else:
            lr = config["init_lr"] * 0.001
        optimizer.param_groups[0]['lr'] = lr

        for i, (index, img, label) in enumerate(cur_dataloader):
            model.train()
            img = img.to(device)
            label_oh = one_hot_label(label, config["n_class"]).to(device)
            label = label.to(device)
            optimizer.zero_grad()
            raw_logits,raw_hash_code, local_logits,local_hash_code,proposalN_windows_logits,window_hash_code \
                = model(img, epoch, i, 'train')
            #全局损失
            L_g1 = criterion(raw_logits, label)  #分类损失
            L_h1 = criterion3(raw_hash_code, label_oh.float(), index)
            #局部损失
            L_g2 = criterion(local_logits, label)  #分类损失
            L_h2 = criterion3(local_hash_code, label_oh.float(), index)
            #part 损失
            L_g3 = criterion(proposalN_windows_logits, label.unsqueeze(1).repeat(1, proposalN).view(-1))

            #total loss
            L_g = L_g1 + L_g2 + L_g3
            L_h = L_h2+L_h1
            B[index, :] = raw_hash_code.data
            B_label[index] = label_oh.data

            if epoch < 2:
                total_loss = L_g
            else:
                total_loss = L_g + L_h
            loss = loss + total_loss
            total_loss.backward()
            optimizer.step()
        #print('epoch:{},bit:{},margin:{},loss:{:.4f},L_g1:{:.4f},L_g2:{:.4f},L_g3:{:.4f},L_h1:{:.4f},L_h2:{:.4f},L_h3:{:.4f}'.format(epoch+1,bit,margin, loss,L_g1,L_g2,L_g3,L_h1,L_h2,L_h3),file=log)
        print('epoch:{},bit:{},margin:{},loss:{:.4f},L_g:{:.4f},L_h:{:.4f}'.format(epoch+1,bit,margin, loss,L_g,L_h))
        loss = loss.cpu().detach().reshape((1)).numpy()
        if not os.path.isdir('loss_result/'+config["dataset"]+'/'+str(bit)):
            os.makedirs('loss_result//'+config["dataset"]+'//'+str(bit))
        with open('loss_result/'+config["dataset"]+'/'+str(bit)+'/'+'loss.txt','a+') as f:
            np.savetxt (f,loss,delimiter= '/n',fmt='%3.5f')
        if (epoch + 1) % config["test_map"] == 0:
            model.snapshot(config["dataset"],epoch, bit)
            model.eval()
            test_code = torch.zeros(num_test, bit)
            test_label_oh = torch.zeros(num_test, config["n_class"])
            train_code = torch.zeros(num_train, bit)
            train_label_oh = torch.zeros(num_train, config["n_class"])
            with torch.no_grad():
                for i, (index, img, label) in enumerate(test_dataloader):
                    img = img.to(device)
                    label_oh = one_hot_label(label, config["n_class"]).to(device)
                    cur_B = model(img, epoch, i, 'test')
                    test_code[index, :] = torch.sign(cur_B).cpu()
                    test_label_oh[index, :] = label_oh.cpu()
                for i, (index, img, label) in enumerate(train_dataloader):
                    img = img.to(device)
                    label_oh = one_hot_label(label, config["n_class"]).to(device)
                    cur_B = model(img, epoch, i, 'test')
                    train_code[index, :] = torch.sign(cur_B).cpu()
                    train_label_oh[index, :] = label_oh.cpu()
            mAP = mean_average_precision(test_code, test_label_oh, train_code, train_label_oh, num_train)
            if mAP > Best_mAP:
                Best_mAP = mAP
            current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
            print('epoch:{},time:{},mAP:{:.4f},Best_mAP:{:.4f}'.format(epoch, current_time, mAP, Best_mAP),file=log)
            print(str(config["info"])+'bit:{},epoch:{},time:{},mAP:{:.4f},Best_mAP:{:.4f}'.format(bit,epoch, current_time, mAP, Best_mAP))
    logger.info(str(bit)+'_map: {:.4f}'.format(mAP)+'_bestmap: {:.4f}'.format(Best_mAP))
    if bit in [16,32,64]:
        if not os.path.isdir('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)):
            os.makedirs('result//'+config["dataset"]+'/'+config["info"]+'//'+str(bit))
        P,R = pr_curve(
            test_code,
            train_code,
            test_label_oh,
            train_label_oh,
        )
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'p.txt',P,fmt='%3.5f')
        np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'r.txt',R,fmt='%3.5f')

if __name__ == '__main__':
    config = get_config()
    print(config)
    logger.add('logs/{dataset}_{info}_{time}.log'.format(
        dataset=str(config["dataset"]),
        info=str(config["info"]),
        time='{time}'
    ))
    for bit in config["bit_list"]:
        for m in config["m"]:
            main(config, bit, m)


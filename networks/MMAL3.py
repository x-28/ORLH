import os
import torch
from torch import nn
import torch.nn.functional as F
from networks import resnet_mm
import numpy as np
from networks.AOLM2 import AOLM
from utils.indices2coordinates import indices2coordinates
from utils.compute_window_nums import compute_window_nums
#####加了nonlinear 非线性投影
stride = 32
channels = 512
input_size = 448
pretrain_path = 'D:/lixue/fine-grained/code/MMAL+sRLH/models/pretrained/resnet18-5c106cde.pth'

if set == 'CUB':
    model_path = './checkpoint/cub'  # pth save path
    root = './datasets/CUB_200_2011'  # dataset path
    num_classes = 200
    # windows info for CUB
    N_list = [2, 3, 2]
    proposalN = sum(N_list)  # proposal window num
    window_side = [128, 192, 256]
    iou_threshs = [0.25, 0.25, 0.25]
    ratios = [[4, 4], [3, 5], [5, 3],
              [6, 6], [5, 7], [7, 5],
              [8, 8], [6, 10], [10, 6], [7, 9], [9, 7], [7, 10], [10, 7]]
else:
    # windows info for CAR and Aircraft
    N_list = [3, 2, 1]
    proposalN = sum(N_list)  # proposal window num
    window_side = [192, 256, 320]
    iou_threshs = [0.25, 0.25, 0.25]
    ratios = [[6, 6], [5, 7], [7, 5],
              [8, 8], [6, 10], [10, 6], [7, 9], [9, 7],
              [10, 10], [9, 11], [11, 9], [8, 12], [12, 8]]
    if set == 'CAR':
        model_path = './checkpoint/car'      # pth save path
        root = './datasets/Stanford_Cars'  # dataset path
        num_classes = 196
    elif set == 'Aircraft':
        model_path = './checkpoint/aircraft'      # pth save path
        root = './datasets/FGVC-aircraft'  # dataset path
        num_classes = 100
'''indice2coordinates'''
window_nums = compute_window_nums(ratios, stride, input_size)
indices_ndarrays = [np.arange(0,window_num).reshape(-1,1) for window_num in window_nums]
coordinates = [indices2coordinates(indices_ndarray, stride, input_size, ratios[i]) for i, indices_ndarray in enumerate(indices_ndarrays)] # 每个window在image上的坐标
coordinates_cat = np.concatenate(coordinates, 0)
window_milestones = [sum(window_nums[:i+1]) for i in range(len(window_nums))]
if set == 'CUB':
    window_nums_sum = [0, sum(window_nums[:3]), sum(window_nums[3:6]), sum(window_nums[6:])]
else:
    window_nums_sum = [0, sum(window_nums[:3]), sum(window_nums[3:8]), sum(window_nums[8:])]

def nms(scores_np, proposalN, iou_threshs, coordinates):
    if not (type(scores_np).__module__ == 'numpy' and len(scores_np.shape) == 2 and scores_np.shape[1] == 1):
        raise TypeError('score_np is not right')

    windows_num = scores_np.shape[0]
    indices_coordinates = np.concatenate((scores_np, coordinates), 1)

    indices = np.argsort(indices_coordinates[:, 0])
    indices_coordinates = np.concatenate((indices_coordinates, np.arange(0,windows_num).reshape(windows_num,1)), 1)[indices]                  #[339,6]
    indices_results = []

    res = indices_coordinates

    while res.any():
        indice_coordinates = res[-1]
        indices_results.append(indice_coordinates[5])

        if len(indices_results) == proposalN:
            return np.array(indices_results).reshape(1,proposalN).astype(np.int)
        res = res[:-1]

        # Exclude anchor boxes with selected anchor box whose iou is greater than the threshold
        start_max = np.maximum(res[:, 1:3], indice_coordinates[1:3])
        end_min = np.minimum(res[:, 3:5], indice_coordinates[3:5])
        lengths = end_min - start_max + 1
        intersec_map = lengths[:, 0] * lengths[:, 1]
        intersec_map[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0
        iou_map_cur = intersec_map / ((res[:, 3] - res[:, 1] + 1) * (res[:, 4] - res[:, 2] + 1) +
                                      (indice_coordinates[3] - indice_coordinates[1] + 1) *
                                      (indice_coordinates[4] - indice_coordinates[2] + 1) - intersec_map)
        res = res[iou_map_cur <= iou_threshs]

    while len(indices_results) != proposalN:
        indices_results.append(indice_coordinates[5])

    return np.array(indices_results).reshape(1, -1).astype(np.int)

class APPM(nn.Module):
    def __init__(self):
        super(APPM, self).__init__()
        self.avgpools = [nn.AvgPool2d(ratios[i], 1) for i in range(len(ratios))]

    def forward(self, proposalN, x, ratios, window_nums_sum, N_list, iou_threshs, DEVICE='cuda'):
        batch, channels, _, _ = x.size()
        avgs = [self.avgpools[i](x) for i in range(len(ratios))]

        # feature map sum
        fm_sum = [torch.sum(avgs[i], dim=1) for i in range(len(ratios))]

        all_scores = torch.cat([fm_sum[i].view(batch, -1, 1) for i in range(len(ratios))], dim=1)
        windows_scores_np = all_scores.data.cpu().numpy()
        window_scores = torch.from_numpy(windows_scores_np).to(all_scores.device).reshape(batch, -1)

        # nms
        proposalN_indices = []
        for i, scores in enumerate(windows_scores_np):
            indices_results = []
            for j in range(len(window_nums_sum)-1):
                indices_results.append(nms(scores[sum(window_nums_sum[:j+1]):sum(window_nums_sum[:j+2])], proposalN=N_list[j], iou_threshs=iou_threshs[j],
                                           coordinates=coordinates_cat[sum(window_nums_sum[:j+1]):sum(window_nums_sum[:j+2])]) + sum(window_nums_sum[:j+1]))
            # indices_results.reverse()
            proposalN_indices.append(np.concatenate(indices_results, 1))   # reverse

        proposalN_indices = np.array(proposalN_indices).reshape(batch, proposalN)
        proposalN_indices = torch.from_numpy(proposalN_indices).to(all_scores.device)
        proposalN_windows_scores = torch.cat(
            [torch.index_select(all_score, dim=0, index=proposalN_indices[i].long()) for i, all_score in enumerate(all_scores)], 0).reshape(
            batch, proposalN)

        return proposalN_indices, proposalN_windows_scores, window_scores

class MainNet(nn.Module):
    def __init__(self, proposalN, num_classes, channels,bit):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        super(MainNet, self).__init__()
        self.num_classes = num_classes
        self.proposalN = proposalN
        self.pretrained_model = resnet_mm.resnet18(pretrained=True, pth_path=pretrain_path)
        self.rawcls_net = nn.Linear(channels, num_classes)
        self.hash_layer=nn.Sequential(
            nn.Linear(num_classes,bit),
            nn.Tanh()###原FISH 是线性变化
        )
        self.APPM = APPM()

    def forward(self, x, epoch, batch_idx, status='test', DEVICE='cuda'):
        fm, embedding, conv5_b = self.pretrained_model(x)
        batch_size, channel_size, side_size, _ = fm.shape
        assert channel_size == 512
        # raw branch
        raw_logits = self.rawcls_net(embedding)
        #更改raw_mask使用raw_logits的device
        raw_mask = torch.ones(raw_logits.size()).detach().to(raw_logits.device) * 0.7#0.7 dog0.3(FF模块)
        #raw_maskz = torch.ones(raw_logits.size()).detach().cuda()
        for i in range(raw_mask.size()[0]):
            raw_mask[i, torch.argmax(raw_logits[i])] = 1
        # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        #raw_logits = raw_logits.to(device)
        #raw_mask = raw_logits.to(raw_logits.device)
        raw_embedings = raw_logits * raw_mask
        raw_hash_code = self.hash_layer(raw_embedings)
        #SCDA
        coordinates = torch.tensor(AOLM(fm.detach(), conv5_b.detach()))#获取左上和右下的坐标

        local_imgs = torch.zeros([batch_size, 3, 448, 448]).to(DEVICE)  # [N, 3, 448, 448]
        for i in range(batch_size):
            [x0, y0, x1, y1] = coordinates[i]
            local_imgs[i:i + 1] = F.interpolate(x[i:i + 1, :, x0:(x1+1), y0:(y1+1)], size=(448, 448),
                                                mode='bilinear', align_corners=True)  # [N, 3, 224, 224]
        local_imgs = local_imgs.to(fm.device)
        local_fm, local_embeddings, _ = self.pretrained_model(local_imgs.detach())  # [N, 2048]
        local_logits = self.rawcls_net(local_embeddings)  # [N, 200]
        local_mask = torch.ones(local_logits.size()).detach().to(local_logits.device) * 0.7#0.7 dog0.3(FF模块)
        for i in range(local_mask.size()[0]):
            local_mask[i, torch.argmax(local_logits[i])] = 1
        local_embeddings = local_logits * local_mask
        local_hash_code = self.hash_layer(local_embeddings)
        proposalN_indices, proposalN_windows_scores, window_scores \
            = self.APPM(self.proposalN, local_fm.detach(), ratios, window_nums_sum, N_list, iou_threshs, DEVICE)

        if status == "train":
            # window_imgs cls
            window_imgs = torch.zeros([batch_size, self.proposalN, 3, 224, 224]).to(DEVICE)  # [N, 4, 3, 224, 224]
            for i in range(batch_size):
                for j in range(self.proposalN):
                    [x0, y0, x1, y1] = coordinates_cat[proposalN_indices[i, j]]
                    window_imgs[i:i + 1, j] = F.interpolate(local_imgs[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)], size=(224, 224),
                                                            mode='bilinear',
                                                            align_corners=True)  # [N, 4, 3, 224, 224]

            window_imgs = window_imgs.reshape(batch_size * self.proposalN, 3, 224, 224)  # [N*4, 3, 224, 224]
            window_imgs = window_imgs.to(fm.device)
            _, window_embeddings, _ = self.pretrained_model(window_imgs.detach())  # [N*4, 2048]
            proposalN_windows_logits = self.rawcls_net(window_embeddings)  # [N* 4, 200]
            windows_mask = torch.ones(proposalN_windows_logits.size()).detach().to(proposalN_windows_logits.device) * 0.7  # 0.7 dog0.3(FF模块)
            for i in range(windows_mask.size()[0]):
                windows_mask[i, torch.argmax(proposalN_windows_logits[i])] = 1

            windows_embedings = proposalN_windows_logits * windows_mask
            window_hash_code = self.hash_layer(windows_embedings)
            # return proposalN_windows_scores, proposalN_windows_logits, proposalN_indices, \
            #        window_scores, coordinates, raw_logits, local_logits, local_imgs,raw_hash_code,local_hash_code
            return raw_logits,raw_hash_code, local_logits,local_hash_code,proposalN_windows_logits,window_hash_code
        else:
            #proposalN_windows_logits = torch.zeros([batch_size * self.proposalN, self.num_classes]).to(DEVICE)

            return local_hash_code
    def snapshot(self,data,iteration,bit):
        torch.save({
            'iteration':iteration,
            'model_state_dict':self.state_dict(),
        },os.path.join('./checkpoint', '{}_model_{}_{}.t'.format(data,iteration,bit)))

    def load_snapshot(self,root):
        checkpoint=torch.load(root)
        self.load_state_dict(checkpoint['model_state_dict'])

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# x = torch.rand(6,3,448,448)
# N_list = [3, 2, 1]
# proposalN = sum(N_list)  # proposal window num
# model = MainNet(proposalN=proposalN, num_classes=100, channels=2048,bit=16)
# model.cuda()
# raw_logits,raw_hash_code, local_logits,local_hash_code,window_hash_code = model(x.cuda(), 1, 0, 'train')
# print(window_hash_code.shape())
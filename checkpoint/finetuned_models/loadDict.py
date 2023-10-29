import torch


s1 = torch.load("../air_epoch146.pth")
s2 = torch.load("air_ft.t")
s3 = s1['model_state_dict']
a = torch.Tensor(64,3,7,7)
a = s3['pretrained_model.conv1.weight']
s1
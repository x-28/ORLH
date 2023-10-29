import torch
from skimage import measure




def AOLM(fms, fm1):
    A = torch.sum(fms, dim=1, keepdim=True)
    a = torch.mean(A, dim=[2, 3], keepdim=True)
    a0 = a*0.8  #自己加的
    # M layer4 后一层 更加关注对象的整体特征 形状、轮廓
    M = (A > a0).float()

    A1 = torch.sum(fm1, dim=1, keepdim=True)
    a1 = torch.mean(A1, dim=[2, 3], keepdim=True)
    a2 = a1 *0.4 #自己加的
    # M layer4 前一层 前景信息 关注大量的细节
    M1 = (A1 > a2).float()


    coordinates = []
    # 为max_idx赋初始值,
    # 解决 UnboundLocalError: local variable 'max_idx' referenced before assignment问题
    #max_idx = 0
    for i, m in enumerate(M):
        mask_np = m.cpu().numpy().reshape(14, 14)
        component_labels = measure.label(mask_np)

        properties = measure.regionprops(component_labels)
        areas = []
        for prop in properties:
            areas.append(prop.area)
        max_idx = areas.index(max(areas))

        #判断两边相加是否=2，相等返回True，不等返回False
        intersection = ((component_labels==(max_idx+1)).astype(int) + (M1[i][0].cpu().numpy()==1).astype(int)) ==2
        #measure.regionprops() 是一个用于计算图像区域属性的函数，可以用于分析二值图像中的连通区域。
        prop = measure.regionprops(intersection.astype(int))
        if len(prop) == 0:
            bbox = [0, 0, 14, 14]
            print('there is one img no intersection')
        else:
            bbox = prop[0].bbox


        x_lefttop = bbox[0] * 32 - 1
        y_lefttop = bbox[1] * 32 - 1
        x_rightlow = bbox[2] * 32 - 1
        y_rightlow = bbox[3] * 32 - 1
        # for image
        if x_lefttop < 0:
            x_lefttop = 0
        if y_lefttop < 0:
            y_lefttop = 0
        coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]
        coordinates.append(coordinate)
    return coordinates

import numpy as np

def mean_average_precision(query_code,query_label,retrieval_code,retrieval_label,R):
    query_code = query_code.cpu().detach().numpy()
    query_label = query_label.cpu().detach().numpy()
    retrieval_code = retrieval_code.cpu().detach().numpy()
    retrieval_label = retrieval_label.cpu().detach().numpy()
    query_num=query_code.shape[0]
    retrieval_code=np.sign(retrieval_code)
    query_code=np.sign(query_code)

    sim=np.dot(retrieval_code,query_code.T)   #矩阵积
    ids=np.argsort(-sim,axis=0)
    APx=[]

    for i in range(query_num):
        label=query_label[i,:]
        label[label==0]=-1   #这里是为了检测是否有相同标签时，不受0的影响
        idx=ids[:,i]
        imatch=np.sum(retrieval_label[idx[0:R],:]==label,axis=1)>0   #retrieval_data是否和这个query有同样的标签。
        relevant_num=np.sum(imatch)
        Lx=np.cumsum(imatch)
        Px=Lx.astype(float)/np.arange(1,R+1,1)
        if relevant_num !=0:
            APx.append(np.sum(Px*imatch)/relevant_num)
        else:
            APx.append(0.0)

    return np.mean(np.array(APx))

def pr_curve(tst_binary, trn_binary, tst_label, trn_label):
    #trn_binary = trn_binary.numpy()
    trn_binary = np.asarray(trn_binary, np.int32)
    trn_label = trn_label.numpy()
    #tst_binary = tst_binary.numpy()
    tst_binary = np.asarray(tst_binary, np.int32)
    tst_label = tst_label.numpy()
    query_times = tst_binary.shape[0]
    trainset_len = trn_binary.shape[0]
    AP = np.zeros(query_times)
    Ns = np.arange(1, trainset_len + 1)



    sum_p = np.zeros(trainset_len)
    sum_r = np.zeros(trainset_len)
    #    f = open("./queryimg.txt", "a+")
    for i in range(query_times):
        #print('Query ', i+1)
        query_label = tst_label[i]
        query_binary = tst_binary[i,:]
        query_result = np.count_nonzero(query_binary != trn_binary, axis=1)    #don't need to divide binary length
        sort_indices = np.argsort(query_result)
        #buffer_yes= np.equal(query_label, trn_label[sort_indices]).astype(int)
        buffer_yes = ((query_label @ trn_label[sort_indices].transpose())>0).astype(float)
        P = np.cumsum(buffer_yes) / Ns
        R = np.cumsum(buffer_yes)/(trainset_len)*10
        sum_p = sum_p+P
        sum_r = sum_r+R

        # print(sort_indices[:10])#每张test图像都会查询一轮，每轮排序都是查询不同的图像的结果。
        # f.writelines(str(sort_indices[:10]))
        # f.write('\r\n')


    return sum_p/query_times,sum_r/query_times







    """
    P-R curve.

    Args
        query_code(torch.Tensor): Query hash code.
        retrieval_code(torch.Tensor): Retrieval hash code.
        query_targets(torch.Tensor): Query targets.
        retrieval_targets(torch.Tensor): Retrieval targets.
        device (torch.device): Using CPU or GPU.

    Returns
        P(torch.Tensor): Precision.
        R(torch.Tensor): Recall.
    """
    # num_query = query_code.shape[0]
    # num_bit = query_code.shape[1]
    # P = torch.zeros(num_query, num_bit + 1).cuda()
    # R = torch.zeros(num_query, num_bit + 1).cuda()
    # for i in range(num_query):
    # gnd = (query_targets[i].unsqueeze(0).mm(retrieval_targets.t()) > 0).float().squeeze().cuda()
    # tsum = torch.sum(gnd)
    # if tsum == 0:
    # continue
    # hamm = 0.5 * (retrieval_code.shape[1] - query_code[i, :] @ retrieval_code.t()).cuda()
    # tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().cuda()).float()
    # total = tmp.sum(dim=-1)
    # total = total + (total == 0).float() * 0.1
    # t = gnd * tmp
    # count = t.sum(dim=-1)
    # p = count / total
    # r = count / tsum
    # P[i] = p
    # R[i] = r
    # mask = (P > 0).float().sum(dim=0)
    # mask = mask + (mask == 0).float() * 0.1
    # P = P.sum(dim=0) / mask
    # R = R.sum(dim=0) / mask




    #return P, R



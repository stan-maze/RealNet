import torch
import torch.nn as nn
from tqdm import tqdm
from utils.misc_helper import to_device
import torch.nn.functional as F
import torch.distributed as dist
import copy, os, cv2
import numpy as np
import torchvision.utils as vutils

class AFS(nn.Module):
    def __init__(self,
                 inplanes,
                 instrides,
                 structure,
                 init_bsn,
                 ):

        super(AFS, self).__init__()

        self.inplanes=inplanes
        self.instrides=instrides
        self.structure=structure
        self.init_bsn=init_bsn

        self.indexes=nn.ParameterDict()

        for block in self.structure:
            for layer in block['layers']:
                self.indexes["{}_{}".format(block['name'],layer['idx'])]=nn.Parameter(torch.zeros(layer['planes']).long(),requires_grad=False)
                self.add_module("{}_{}_upsample".format(block['name'],layer['idx']),
                                nn.UpsamplingBilinear2d(scale_factor=self.instrides[layer['idx']]/block['stride']))


    @torch.no_grad()
    def forward(self, inputs,train=False):
        block_feats = {}
        feats = inputs["feats"]
        for block in self.structure:
            block_feats[block['name']]=[]

            for layer in block['layers']:
                feat_c=torch.index_select(feats[layer['idx']]['feat'], 1, self.indexes["{}_{}".format(block['name'],layer['idx'])].data)
                feat_c=getattr(self,"{}_{}_upsample".format(block['name'],layer['idx']))(feat_c)
                block_feats[block['name']].append(feat_c)
            block_feats[block['name']]=torch.cat(block_feats[block['name']],dim=1)

        if train:
            gt_block_feats = {}
            gt_feats = inputs["gt_feats"]
            for block in self.structure:
                gt_block_feats[block['name']] = []
                for layer in block['layers']:
                    feat_c = torch.index_select(gt_feats[layer['idx']]['feat'], 1, self.indexes["{}_{}".format(block['name'], layer['idx'])].data)
                    feat_c = getattr(self, "{}_{}_upsample".format(block['name'], layer['idx']))(feat_c)
                    gt_block_feats[block['name']].append(feat_c)
                gt_block_feats[block['name']] = torch.cat(gt_block_feats[block['name']], dim=1)
            return {'block_feats':block_feats,"gt_block_feats":gt_block_feats}

        return {'block_feats':block_feats}



    def get_outplanes(self):
        return { block['name']:sum([layer['planes'] for layer in block['layers']])  for block in self.structure}

    def get_outstrides(self):
        return { block['name']:block['stride']  for block in self.structure}


    @torch.no_grad()
    def init_idxs(self, model, train_loader, distributed=True):
        anomaly_types = copy.deepcopy(train_loader.dataset.anomaly_types)

        if 'normal' in train_loader.dataset.anomaly_types:
            del train_loader.dataset.anomaly_types['normal']

        for key in train_loader.dataset.anomaly_types:
            train_loader.dataset.anomaly_types[key] = 1.0/len(list(train_loader.dataset.anomaly_types.keys()))

        model.eval()
        criterion = nn.MSELoss(reduce=False).to(model.device)
        for block in self.structure:
            self.init_block_idxs(block, model, train_loader, criterion,distributed=distributed)
        train_loader.dataset.anomaly_types = anomaly_types
        model.train()


    def init_block_idxs(self,block,model,train_loader,criterion,distributed=True):

        if distributed:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            if rank == 0:
                tq = tqdm(range(self.init_bsn), desc="init {} index".format(block['name']))
            else:
                tq = range(self.init_bsn)
        else:
            tq = tqdm(range(self.init_bsn), desc="init {} index".format(block['name']))

        cri_sum_vec=[torch.zeros(self.inplanes[layer['idx']]).to(model.device) for layer in block['layers']]
        iterator = iter(train_loader)

        for bs_i in tq:
            try:
                input = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                input = next(iterator)
        

            bb_feats = model.backbone(to_device(input),train=True)

            ano_feats=bb_feats['feats']
            ori_feats=bb_feats['gt_feats']
            gt_mask = input['mask'].to(model.device)

            B= gt_mask.size(0)

            ori_layer_feats=[ori_feats[layer['idx']]['feat'] for layer in block['layers']]
            ano_layer_feats=[ano_feats[layer['idx']]['feat'] for layer in block['layers']]

            for i,(ano_layer_feat,ori_layer_feat) in enumerate(zip(ano_layer_feats,ori_layer_feats)):
                layer_name=block['layers'][i]['idx']

                C = ano_layer_feat.size(1)

                ano_layer_feat = getattr(self, "{}_{}_upsample".format(block['name'], layer_name))(ano_layer_feat)
                ori_layer_feat = getattr(self, "{}_{}_upsample".format(block['name'], layer_name))(ori_layer_feat)

                layer_pred = (ano_layer_feat - ori_layer_feat) ** 2

                _, _, H, W = layer_pred.size()

                layer_pred = layer_pred.permute(1, 0, 2, 3).contiguous().view(C, B * H * W)
                (min_v, _), (max_v, _) = torch.min(layer_pred, dim=1), torch.max(layer_pred, dim=1)
                layer_pred = (layer_pred - min_v.unsqueeze(1)) / (max_v.unsqueeze(1) - min_v.unsqueeze(1)+ 1e-4)

                label = F.interpolate(gt_mask, (H, W), mode='nearest')
                label = label.permute(1, 0, 2, 3).contiguous().view(1, B * H * W).repeat(C, 1)

                mse_loss = torch.mean(criterion(layer_pred, label), dim=1)

                if distributed:
                    mse_loss_list = [mse_loss for _ in range(world_size)]
                    dist.all_gather(mse_loss_list, mse_loss)
                    mse_loss = torch.mean(torch.stack(mse_loss_list,dim=0),dim=0,keepdim=False)

                cri_sum_vec[i] += mse_loss

        for i in range(len(cri_sum_vec)):
            cri_sum_vec[i][torch.isnan(cri_sum_vec[i])] = torch.max(cri_sum_vec[i][~torch.isnan(cri_sum_vec[i])])
            values, indices = torch.topk(cri_sum_vec[i], k=block['layers'][i]['planes'], dim=-1, largest=False)
            values, _ = torch.sort(indices)
            # print(values)

            if distributed:
                tensor_list = [values for _ in range(world_size)]
                dist.all_gather(tensor_list, values)
                self.indexes["{}_{}".format(block['name'], block['layers'][i]['idx'])].data.copy_(tensor_list[0].long())
            else:
                self.indexes["{}_{}".format(block['name'], block['layers'][i]['idx'])].data.copy_(values.long())
    

        # # 可视化残差
        # for bs_i in tq:
        #     try:
        #         input = next(iterator)
        #     except StopIteration:
        #         iterator = iter(train_loader)
        #         input = next(iterator)

        #     # 假设 input['image'] 是包含输入图像的键
        #     input_images = input['image']  # 形状为 (batch_size, channels, height, width)
        #     input_masks = input['mask']  # 形状为 (batch_size, channels, height, width)
        #     save_dir = "residual_images"
        #     os.makedirs(save_dir, exist_ok=True)
        #     # 保存每个 batch 中的图像
        #     for i in range(input_images.size(0)):  # 遍历 batch
        #         img_tensor = input_images[i]  # 提取第 i 个图像
        #         mask_tensor = input_masks[i]
        #         save_path = os.path.join(save_dir, f"input_{bs_i}_{i}.png")
        #         vutils.save_image(img_tensor, save_path)  # 保存为 PNG 文件
        #         save_path = os.path.join(save_dir, f"mask_{bs_i}_{i}.png")
        #         vutils.save_image(mask_tensor, save_path)  # 保存为 PNG 文件
        #     bb_feats = model.backbone(to_device(input),train=True)
        #     layer = 1
        #     self.save_residual_features(bb_feats, output_dir=f"{save_dir}B_128_256_128_64/block{layer}", block_name=f"block{layer}", layer_idx=f'layer{layer}')
        #     layer = 2
        #     self.save_residual_features(bb_feats, output_dir=f"{save_dir}B_128_256_128_64/block{layer}", block_name=f"block{layer}", layer_idx=f'layer{layer}')
        #     layer = 3
        #     self.save_residual_features(bb_feats, output_dir=f"{save_dir}B_128_256_128_64/block{layer}", block_name=f"block{layer}", layer_idx=f'layer{layer}')
        #     layer = 4
        #     self.save_residual_features(bb_feats, output_dir=f"{save_dir}B_128_256_128_64/block{layer}", block_name=f"block{layer}", layer_idx=f'layer{layer}')
        #     exit()

    

    @torch.no_grad()
    def save_residual_features(self, inputs, output_dir, block_name=None, layer_idx=None):
        """
        根据 self.indexes 挑选的索引保存残差特征为图像.
        
        Args:
            inputs (dict): 包含 'feats' 和 'gt_feats' 的字典.
            output_dir (str): 保存图像的输出目录.
            block_name (str, optional): 指定块的名称，如果为 None，则保存所有块的特征.
            layer_idx (int, optional): 指定层的索引，如果为 None，则保存所有层的特征.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        feats = inputs["feats"]
        gt_feats = inputs["gt_feats"]

        # 遍历结构
        for block in self.structure:
            if block_name is not None and block['name'] != block_name:
                continue  # 跳过非目标块

            print(block['name'])
            for layer in block['layers']:
                print(layer)
                if layer_idx is not None and layer['idx'] != layer_idx:
                    continue  # 跳过非目标层

                # 从 self.indexes 中获取选中的特征索引
                selected_idx = self.indexes["{}_{}".format(block['name'], layer['idx'])].data
                if selected_idx.numel() == 0:
                    continue  # 如果没有索引，跳过

                # 获取残差特征
                feat_c = torch.index_select(feats[layer['idx']]['feat'], 1, selected_idx)
                gt_feat_c = torch.index_select(gt_feats[layer['idx']]['feat'], 1, selected_idx)
                residual_feat = feat_c - gt_feat_c  # 计算残差

                # 取出批次大小和通道数
                B, C, H, W = residual_feat.shape
                residual_feat = residual_feat.cpu().numpy()  # 转换为 NumPy 数组

                # 保存每个通道的残差图像
                for b in range(B):
                    for c in range(C):
                        img = residual_feat[b, c, :, :]
                        img = (img - img.min()) / (img.max() - img.min()) * 255  # 归一化到 0-255
                        img = img.astype(np.uint8)  # 转换为 uint8 格式

                        # 构建文件名
                        file_name = f"{block['name']}_layer{layer['idx']}_batch{b}_channel{c}.png"
                        file_path = os.path.join(output_dir, file_name)

                        # 保存图片
                        cv2.imwrite(file_path, img)

                        print(f"Saved: {file_path}")


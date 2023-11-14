import os
import cv2
import pyvips
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from .utils import get_concentrations, percentile, TissueMaskException
from .stain_extractor_cpu import StainExtractorCPU
from .stain_extractor_gpu import StainExtractorGPU
from tqdm import tqdm
from einops import rearrange
from utils.utils import standardize


class TorchVahadaneNormalizer():
    """
    Source code adapted from:
    https://github.com/Peter554/StainTools/blob/master/staintools/stain_normalizer.py
    Uses StainTools as dependency could be changed by integrating VahadaneStainExtractor directly.
    if direct usage on gpu, idea is possibility to use nvidia based loadings with gpu decompression of wsi
    """

    def __init__(self, device='cuda', staintools_estimate=True):
        super().__init__()
        self.maxC_target = None
        self.maxCs = {}
        self.method = 'ista'
        self.device = torch.device(device)
        self.staintools_estimate = staintools_estimate
        if self.staintools_estimate:
            self.stain_extractor = StainExtractorCPU()
        else:
            self.stain_extractor = StainExtractorGPU()

    def get_coefs(self, I):

        if self.staintools_estimate:
            stain_matrix = self.stain_extractor.get_stain_matrix(
                I).astype(np.float32)
            stain_matrix = torch.from_numpy(stain_matrix).to(self.device)
            I = torch.from_numpy(I).to(self.device)
        else:
            if not (type(I) == torch.Tensor):
                I = torch.from_numpy(I).to(self.device)
            stain_matrix = self.stain_extractor.get_stain_matrix(I)

        concentrations = get_concentrations(
            I, stain_matrix, method=self.method)
        maxC = percentile(concentrations.T, 99, dim=0)
        return stain_matrix.cpu().numpy(), maxC.cpu().numpy()  # type: ignore

    def fit(self, slide_name, dataloader: DataLoader, p, div=3):
        stain_mats = []
        max_cs = []
        for n, (data, coor, size) in tqdm(enumerate(dataloader)):
            if n > (len(dataloader) // div):
                break
            data = rearrange(data, 'n h w d -> (n h) w d')
            coor = coor.numpy()
            size = size.numpy()
            data = data.numpy()
            try:
                data, _ = standardize(data, p=p)
                stain_mat, max_c = self.get_coefs(data)
                stain_mats.append(stain_mat)
                max_cs.append(max_c)
            except TissueMaskException:
                pass
        stain_mat = torch.from_numpy(
            np.median(np.array(stain_mats), axis=0)).to(self.device)
        max_c = torch.from_numpy(
            np.median(np.array(max_cs), axis=0)).to(self.device)
        # self.stain_m_fixed = torch.from_numpy(np.percentile(np.array(stain_mats), 50,  axis=0)).to(torch.float32).to(self.device)
        # self.maxC = torch.from_numpy(np.percentile(np.array(max_cs), 50, axis=0)).to(torch.float32).to(self.device)

        raise NotImplementedError
        # TODO: 删除了部分代码：1. fit这里不应该计算tranform过程的东西，2. 需要apply的数据计算的中间变量更是不应该存在self里面，
        #  而且后续可能会需要考虑一种情况是把normalizer以pkl方式保存下来以备直接读取然后apply到对应的数据
        self.stain_matrix_target = stain_mat
        self.maxC_target = max_c
        print(self.stain_matrix_target.cpu(), 'set as target stain matrix')
        print('max concentration of target:  ', self.maxC_target.cpu())

    def transform_(self, I, slide_name):

        if not (type(I) == torch.Tensor):
            I = torch.from_numpy(I).to(self.device)

        concentrations = get_concentrations(I, self.stain_matrices[slide_name], method=self.method)
        concentrations *= (self.maxC_target / self.maxCs[slide_name])[:, None]

        out = 255 * torch.exp(-1 * torch.matmul(concentrations.T,
                                                self.stain_matrix_target))  # type: ignore
        return out.reshape(I.shape).type(torch.uint8)

    def transform(self, slide_name, dataloader: DataLoader, p, new_slide):
        raise NotImplementedError
        # TODO: 如果按照你的api目前的设计，stain_mat, max_c应该在这里才开始计算才对。
        # 如果你想改得更好一点，其实这部分可以在其他地方进行。按理说，transform/fit这两个函数只需要负责简单的计算，而读取数据、保存数据应该是别的地方负责的
        stain_mat = self.stain_matrices[slide_name]
        max_c = self.maxCs[slide_name]
        print(self.stain_matrix_target, '(target stain matrix)')
        print(stain_mat, 'set as transform matrix')
        print('maxC_target: ', self.maxC_target.cpu())  # type: ignore
        print('maxC:        ', max_c.cpu())
        print('align_ratio: ', (self.maxC_target / max_c).cpu())

        for data, coor, size in tqdm(dataloader):
            h = data.size()[1]
            data = rearrange(data, 'n h w d -> (n h) w d')
            coor = coor.numpy()
            size = size.numpy()
            data = data.numpy()
            data, _ = standardize(data, p=p)
            tpat = self.transform_(data, slide_name)
            tpat = tpat.cpu() if isinstance(tpat, torch.Tensor) else tpat
            tpat = rearrange(tpat, '(n h) w d -> n h w d', h=h)
            for n, (x, y) in enumerate(coor):
                sizex, sizey = size[n]
                new_slide[y:y + sizey, x:x + sizex] = tpat[n][:sizey, :sizex]

        vi = pyvips.Image.new_from_array(new_slide)
        fmt = slide_name.split('.')[-1]

        raise NotImplementedError
        # TODO@DK(20231114): 如果这里要这么写，应该是把cfg的东西通过参数传进来，但是其实尽量应该避免传一个cfg这样的东西，
        #  因为其他人会看不懂，以及相当于是一种与其他项目过度的耦合，最好是每个参数逐个传，比如save_dir, source ...
        # save_path = os.path.join(cfg.save_dir, slide_name.replace(fmt, 'tif'))
        # vi.tiffsave(save_path, tile=True, compression='jpeg',
        #             bigtiff=True, pyramid=True)  # type: ignore
        #
        # if cfg.get_thumb:
        #     quick_check_path = os.path.join(
        #         cfg.source, 'quick_check', slide_name.replace('.' + fmt, '_thumbn_cn.png'))
        #     thumbn = Image.fromarray(new_slide)
        #     thumbn.thumbnail((4096, 4096), Image.LANCZOS)
        #     thumbn = np.array(thumbn)[:, :, :3].astype(np.uint8)
        #     cv2.imwrite(quick_check_path, cv2.cvtColor(
        #         thumbn, cv2.COLOR_BGR2RGB))

        print('__________________________________________________')

# TODO: 这里是不需要这样写的， if __name__ == '__main__' 一般是放在程序入口，或者单独测试某个文件用的。
# if __name__ == '__main__':
#     print('haha!')

import os
import cv2
import pyvips
import torch
import numpy as np
from PIL import Image
from .utils import get_concentrations, percentile, TissueMaskException
from .stain_extractor_cpu import StainExtractorCPU
from .stain_extractor_gpu import StainExtractorGPU
from tqdm import tqdm
from einops import rearrange
from utils.utils import standardize
from config import Config
cfg = Config()


class TorchVahadaneNormalizer():
    """
    Source code adapted from:
    https://github.com/Peter554/StainTools/blob/master/staintools/stain_normalizer.py
    Uses StainTools as dependency could be changed by integrating VahadaneStainExtractor directly.
    if direct usage on gpu, idea is possibility to use nvidia based loadings with gpu decompression of wsi
    """

    def __init__(self, device='cuda', staintools_estimate=True):
        super().__init__()
        self.stain_matrix_target = None
        self.stain_matrices = {}
        self.maxC_target = None
        self.maxCs = {}
        self.method = 'ista'
        self.device = torch.device(device)
        self.staintools_estimate = staintools_estimate
        if self.staintools_estimate:
            self.stain_extractor = StainExtractorCPU()
        else:
            self.stain_extractor = StainExtractorGPU()

    def fit_(self, I):

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

    def fit(self, slide_name, dataloader, p, div=3):
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
                stain_mat, max_c = self.fit_(data)
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

        if self.stain_matrix_target == None:
            self.stain_matrix_target = stain_mat
            self.maxC_target = max_c
            print(self.stain_matrix_target.cpu(), 'set as target stain matrix')
            print('max concentration of target:  ', self.maxC_target.cpu())
        else:
            self.stain_matrices[slide_name] = stain_mat
            self.maxCs[slide_name] = max_c
        return

    def transform_(self, I, slide_name):

        if not (type(I) == torch.Tensor):
            I = torch.from_numpy(I).to(self.device)

        concentrations = get_concentrations(
            I, self.stain_matrices[slide_name], method=self.method)
        concentrations *= (self.maxC_target / self.maxCs[slide_name])[:, None]

        out = 255 * torch.exp(-1 * torch.matmul(concentrations.T,
                              self.stain_matrix_target))  # type: ignore
        return out.reshape(I.shape).type(torch.uint8)

    def transform(self, slide_name, dataloader, p, new_slide):
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
        save_path = os.path.join(cfg.save_dir, slide_name.replace(fmt, 'tif'))
        vi.tiffsave(save_path, tile=True, compression='jpeg',
                    bigtiff=True, pyramid=True)  # type: ignore

        if cfg.get_thumb:
            quick_check_path = os.path.join(
                cfg.source, 'quick_check', slide_name.replace('.' + fmt, '_thumbn_cn.png'))
            thumbn = Image.fromarray(new_slide)
            thumbn.thumbnail((4096, 4096), Image.LANCZOS)
            thumbn = np.array(thumbn)[:, :, :3].astype(np.uint8)
            cv2.imwrite(quick_check_path, cv2.cvtColor(
                thumbn, cv2.COLOR_BGR2RGB))

        print('__________________________________________________')


if __name__ == '__main__':
    print('haha!')

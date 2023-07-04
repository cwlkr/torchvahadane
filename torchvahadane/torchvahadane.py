from torchvahadane.optimizers import ista, coord_descent
import torch
import numpy as np
from torchvahadane.dict_learning import dict_learning
from torchvahadane.utils import get_concentrations, convert_OD_to_RGB, percentile
from torchvahadane.stain_extractor_cpu import StainExtractorCPU
from torchvahadane.stain_extractor_gpu import StainExtractorGPU

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
        self.maxC_target = None
        self.stain_m_fixed = None
        self.device = torch.device(device)
        self.staintools_estimate = staintools_estimate
        if self.staintools_estimate:
            self.stain_extractor = StainExtractorCPU()
        else:
            self.stain_extractor =  StainExtractorGPU()

    def fit(self, target, method='ista'):
        # target = target.astype("float32")
        if self.staintools_estimate:
            self.stain_matrix_target = self.stain_extractor.get_stain_matrix(target).astype(np.float32)
            self.stain_matrix_target = torch.from_numpy(self.stain_matrix_target).to(self.device)
            target = torch.from_numpy(target).to(self.device)
        else:
            if not (type(target) == torch.Tensor):
                target = torch.from_numpy(target)
            self.stain_matrix_target = self.stain_extractor.get_stain_matrix(target.to(self.device))
        concentration_target = get_concentrations(target, self.stain_matrix_target,  method = method)
        self.maxC_target = percentile(concentration_target.T, 99, dim=0)


    def set_transform_stain_matrix(self, I):
        """there could also be a way of fixing a target matrix
        If a set of transformations are done on same wsi, stain matrix should not meaningfully change over time,
        have a moving average and if average converges to new samples then set as matrix?"""

        if self.staintools_estimate:
            self.stain_m_fixed  = self.stain_extractor.get_stain_matrix(I).astype(np.float32)
            self.stain_m_fixed  = torch.from_numpy(self.stain_m_fixed ).to(self.device)
        else:
            if not (type(I) == torch.Tensor):
                I = torch.from_numpy(I)
            self.stain_m_fixed = self.stain_extractor.get_stain_matrix(I.to(self.device))
        print(self.stain_m_fixed, 'set as transform matrix')


    def transform(self, I, method='ista'):
        ## add option to skip matrix calculation
        if self.stain_m_fixed is None:
            if self.staintools_estimate:
                stain_matrix = self.stain_extractor.get_stain_matrix(I).astype(np.float32)
                stain_matrix = torch.from_numpy(stain_matrix).to(self.device)
                I = torch.from_numpy(I).to(self.device)
            else:
                if not (type(I) == torch.Tensor):
                    I = torch.from_numpy(I).to(self.device)
                stain_matrix = self.stain_extractor.get_stain_matrix(I)
        else:
            if not (type(I) == torch.Tensor):
                I = torch.from_numpy(I).to(self.device)
            stain_matrix = self.stain_m_fixed

        concentrations = get_concentrations(I, stain_matrix, method = method)
        maxC = percentile(concentrations.T, 99, dim=0)
        concentrations *= (self.maxC_target / maxC)[:,None]
        out = 255 * torch.exp(-1 * torch.matmul(concentrations.T, self.stain_matrix_target))
        return out.reshape(I.shape).type(torch.uint8)


if __name__ == '__main__':

    import cv2
    import matplotlib.pyplot as plt
    target = cv2.imread('test_images/TCGA-33-4547-01Z-00-DX7.91be6f90-d9ab-4345-a3bd-91805d9761b9_8270_5932_0.png')
    target = cv2.cvtColor(target, 4)
    norm = cv2.imread('test_images/TCGA-95-8494-01Z-00-DX1.716299EF-71BB-4095-8F4D-F0C2252CE594_5932_5708_0.png')
    norm = cv2.cvtColor(norm, 4)

    # test implementation
    n = TorchVahadaneNormalizer()
    n.fit(target)
    normalized_img = n.transform(norm)
    plt.imshow(normalized_img.cpu())
    plt.imshow(norm)

    ## stain tools
    import staintools
    n = staintools.StainNormalizer('vahadane')
    n.fit(target)
    normalized_img = n.transform(norm)
    plt.imshow(normalized_img)
    plt.imshow(norm)

    # test implementation gpu only
    norm = torch.from_numpy(norm)
    target = torch.from_numpy(target)
    n = TorchVahadaneNormalizer(staintools_estimate=False)
    n.fit(target)
    normalized_img = n.transform(norm)
    plt.imshow(normalized_img.cpu())

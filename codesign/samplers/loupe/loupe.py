from .loupe_modules import LineConstrainedProbMask, ProbMask, ProbMaskMutliAccn
from ..modules import NanDebugger, preselect, RescaleProbMap, ThresholdRandomMaskSigmoidV1, MaximumBinarize, MaximumBinarizeLineConstrained
import torch.nn as nn
import torch 

class LOUPELikeSampler(nn.Module):
    def __init__(
        self, 
        shape=[320, 320], 
        subsampling_dim=[-2, -1],
        acceleration=4, 
        line_constrained=False,
        preselect_num=0,
        preselect_ratio=0
    ):
        super().__init__()
        
        # properties
        self.shape = shape
        self.subsampling_dim = subsampling_dim
        self.acceleration = acceleration
        self.line_constrained = line_constrained
        self.preselect_num = preselect_num
        self.preselect_ratio = preselect_ratio

        # operations
        self.rescale = RescaleProbMap
        self.binarize = ThresholdRandomMaskSigmoidV1.apply # FIXME
        self.get_probability_mask = None

    @property
    def preselect(self):
        return self.preselect_num > 0 or self.preselect_ratio > 0

    @property
    def sampler_budget(self):
        if self.preselect_ratio == 0:
            return 1 / self.acceleration
        else:
            return 1 / self.acceleration - 1 / self.preselect_ratio

    def forward(self, kspace):
        mask_prob = self.get_probability_mask(kspace)

        # set preselect location to probability 0 for avoiding rescaling
        if self.preselect:
            mask_prob = preselect(
                mask_prob,
                dim=self.subsampling_dim, 
                preselect_num=self.preselect_num, 
                preselect_ratio=self.preselect_ratio, 
                value=0, # set preselect region to be 0
                line_constrained=self.line_constrained
            )

        mask_rescaled = self.rescale(mask_prob, self.sampler_budget) 

        if self.training:
            mask_binarized = self.binarize(mask_rescaled)
        else:
            if self.line_constrained:
                mask_binarized = MaximumBinarizeLineConstrained(mask_rescaled)
            else:
                mask_binarized = MaximumBinarize(mask_rescaled)

        # preselect
        if self.preselect:
            mask_binarized = preselect(
                mask_binarized,
                dim=self.subsampling_dim, 
                preselect_num=self.preselect_num, 
                preselect_ratio=self.preselect_ratio, 
                value=1, # set preselect region to be 1
                line_constrained=self.line_constrained
            )

        if mask_binarized.dim() != kspace.dim():
            # multicoil data 
            mask_binarized = mask_binarized.unsqueeze(1)

        kspace_masked = mask_binarized * kspace
        mask_binarized = mask_binarized.squeeze(1)
        self.mask_binarized_vis = mask_binarized.cpu().detach()[0]
        
        return kspace_masked, mask_binarized

class LOUPESampler(LOUPELikeSampler):
    def __init__(
        self, 
        shape=[320, 320], 
        subsampling_dim=[-2, -1],
        slope=5, 
        acceleration=4, 
        line_constrained=False,
        preselect_num=0,
        preselect_ratio=0
    ):
        super().__init__(
            shape, 
            subsampling_dim,
            acceleration, 
            line_constrained,
            preselect_num,
            preselect_ratio
        )

        # probability mask
        if line_constrained:
            assert subsampling_dim == -1, "subsampling_dim must be -1 when line_constrained is True"
            self.get_probability_mask = LineConstrainedProbMask(
                shape=shape, 
                slope=slope
            )
        else:
            self.get_probability_mask = ProbMask(
                shape=shape, 
                slope=slope, 
                preselect=self.preselect, 
                preselect_num=preselect_num, 
                preselect_ratio=preselect_ratio
            )

class LoupeSamplerMultiAcceleration(nn.Module):
    def __init__(
        self, 
        shape=[320, 320], 
        subsampling_dim=[-2, -1],
        slope=5,
        accelerations=[4,8,16,32,64],
        line_constrained=False,
        preselect_nums=None,
        preselect_ratios=[32, 64, 128, 256, 512]
    ):
        super().__init__()
        
        # properties
        self.shape = shape
        self.subsampling_dim = subsampling_dim
        self.accelerations = accelerations
        self.line_constrained = line_constrained
        self.preselect_nums = preselect_nums 
        self.slope = slope
        self.preselect_ratios = preselect_ratios

        # operations
        self.rescale = RescaleProbMap
        self.binarize = ThresholdRandomMaskSigmoidV1.apply # FIXME
        self.get_probability_mask = None

        if line_constrained:
            assert subsampling_dim == -1, "subsampling_dim must be -1 when line_constrained is True"
            self.get_probability_mask = LineConstrainedProbMask(
                shape=shape, 
                slope=slope
            )
        else:
            self.get_probability_mask = ProbMaskMutliAccn(
                shape=shape, 
                slope=slope, 
                # preselect=self.preselect, 
                # preselect_num=preselect_nums, 
                # preselect_ratio=preselect_ratios
            )

    @property
    def preselect(self):
        return self.preselect_nums is not None or self.preselect_ratios is not None

    def sampler_budget(self, ind):
        if self.preselect_ratios == None:
            return 1 / self.accelerations[ind]
        else:
            return 1 / self.accelerations[ind] - 1 / self.preselect_ratios[ind]

    def forward(self, kspace):
        # randomly choose an acceleration factor
        acc_tensor = torch.tensor(self.accelerations)
        acc_ind = torch.randint(0, len(acc_tensor), (1,))

        sample_acceleration = acc_tensor[torch.randint(0, len(acc_tensor), (1,))].item()
        sample_preselect_ratio = self.preselect_ratios[acc_ind] if self.preselect_ratios is not None else 0
        
        mask_prob = self.get_probability_mask(kspace)

        # set preselect location to probability 0 for avoiding rescaling
        if self.preselect:
            mask_prob = preselect(
                mask_prob,
                dim=self.subsampling_dim, 
                preselect_num=self.preselect_num[acc_ind] if self.preselect_nums is not None else 0, 
                preselect_ratio=sample_preselect_ratio,
                value=0, # set preselect region to be 0
                line_constrained=self.line_constrained
            )

        mask_rescaled = self.rescale(mask_prob, self.sampler_budget(acc_ind)) 

        if self.training:
            mask_binarized = self.binarize(mask_rescaled)
        else:
            if self.line_constrained:
                mask_binarized = MaximumBinarizeLineConstrained(mask_rescaled)
            else:
                mask_binarized = MaximumBinarize(mask_rescaled)

        # preselect
        if self.preselect:
            mask_prob = preselect(
                mask_prob,
                dim=self.subsampling_dim, 
                preselect_num=self.preselect_num[acc_ind] if self.preselect_nums is not None else 0,
                preselect_ratio=sample_preselect_ratio,
                value=1, # set preselect region to be 0
                line_constrained=self.line_constrained
            )

        if mask_binarized.dim() != kspace.dim():
            # multicoil data 
            mask_binarized = mask_binarized.unsqueeze(1)

        kspace_masked = mask_binarized * kspace
        mask_binarized = mask_binarized.squeeze(1)
        self.mask_binarized_vis = mask_binarized.cpu().detach()[0]
        
        return kspace_masked, mask_binarized, sample_preselect_ratio

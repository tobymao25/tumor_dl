import os
import cv2
import torch
import datetime
from monai.data import decollate_batch
from monai.inferers import SimpleInferer
from monai.transforms import (
    Compose,
    Resize,
    ToTensor,
    AddChannel,
    LoadImage,
    AsChannelFirst,
    NormalizeIntensity,
    Orientation,
    Spacing,
    Activations,
    AsDiscrete
)
import nibabel as nib 
# Additional Scripts
from train_unetr import UneTRSeg
from utils import create_folder_if_not_exist

from config import cfg


class SegInference:
    inferer = SimpleInferer()

    def __init__(self, model_path, device):
        self.device = device
        self.img_dim = cfg.unetr.img_dim

        self.unetr = UneTRSeg(device)
        self.unetr.load_model(model_path)
        self.unetr.model.eval()

        create_folder_if_not_exist('./results')

    def infer(self, path, save=True):

        data = self.preprocess(path)
        with torch.no_grad():
            pred_mask = self.inferer(inputs=data, network=self.unetr.model)
            pred_mask = self.postprocess(pred_mask)

        if save:
            self.save_masks(path, pred_mask)

        return pred_mask

    def preprocess(self, path):
        transform = Compose(
            [
                LoadImage(image_only=True),
                ToTensor(),
                AsChannelFirst(),
                Orientation(axcodes="RAS", image_only=True),
                Spacing(
                    pixdim=(1.0, 1.0, 1.0),
                    mode="bilinear",
                    image_only=True
                ),
                Resize(spatial_size=cfg.unetr.img_dim, mode='nearest'),
                NormalizeIntensity(nonzero=True, channel_wise=True),
                AddChannel(),
            ]
        )

        data = transform(path).to(self.device)
        return data

    def postprocess(self, pred_mask):
        transform = Compose(
            [
                Activations(sigmoid=True),
                AsDiscrete(threshold=0.5)
            ]
        )

        pred_mask = [transform(i).cpu().detach().numpy() for i in decollate_batch(pred_mask)]
        return pred_mask

    def save_masks(self, path, pred_mask):
        name = path.split('/')[-1].split('.')[0]
        create_folder_if_not_exist(os.path.join('./results', name))
        create_folder_if_not_exist(os.path.join('./results', name, 'TC'))
        create_folder_if_not_exist(os.path.join('./results', name, 'WT'))
        create_folder_if_not_exist(os.path.join('./results', name, 'ET'))
        base_path = os.path.join('./results', name)
        TC_path = os.path.join('./results', name, 'TC')
        WT_path = os.path.join('./results', name, 'WT')
        ET_path = os.path.join('./results', name, 'ET')
        """
        for idx in range(self.img_dim[-1]):
            cv2.imwrite(os.path.join('./results', name, 'TC', f'{idx}.png'),
                        pred_mask[0][0][..., idx] * 255)

            cv2.imwrite(os.path.join('./results', name, 'WT', f'{idx}.png'),
                        pred_mask[0][1][..., idx] * 255)

            cv2.imwrite(os.path.join('./results', name, 'ET', f'{idx}.png'),
                        pred_mask[0][2][..., idx] * 255)"""
        img = nib.load(path)
        affine = img.affine
        tc_nifti = nib.Nifti1Image(pred_mask[0][0], affine)
        wt_nifti = nib.Nifti1Image(pred_mask[0][1], affine)
        et_nifti = nib.Nifti1Image(pred_mask[0][2], affine)
        nib.save(tc_nifti, os.path.join(TC_path, f'{name}_TC.nii.gz'))
        nib.save(wt_nifti, os.path.join(WT_path, f'{name}_WT.nii.gz'))
        nib.save(et_nifti, os.path.join(ET_path, f'{name}_ET.nii.gz'))
        TC_seg_path = os.path.join(TC_path, f'{name}_TC.nii.gz')
        WT_seg_path = os.path.join(WT_path, f'{name}_WT.nii.gz')
        ET_seg_path = os.path.join(ET_path, f'{name}_ET.nii.gz')
        return TC_seg_path, WT_seg_path, ET_seg_path
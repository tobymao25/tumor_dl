import os
import cv2
import torch
import datetime
import numpy as np
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
from nibabel.orientations import axcodes2ornt, ornt_transform, apply_orientation, aff2axcodes
from config import cfg
import SimpleITK as sitk
from PIL import Image, ImageDraw, ImageFont
import scipy.ndimage
from scipy.ndimage import binary_fill_holes

def resize(affine, pm, nifti_path):
    desired_shape = (240, 240, 155)
    zoom_factors = [d / o for d, o in zip(desired_shape, pm.shape)]
    resized_mask = scipy.ndimage.zoom(pm, zoom_factors, order=1) 
    filled_mask = binary_fill_holes(resized_mask)
    filled_mask = np.flip(filled_mask, axis=0) 
    filled_mask = filled_mask.astype(np.int32)
    print("Resized shape:", filled_mask.shape)
    new_img = nib.Nifti1Image(filled_mask, affine)
    new_img_np = new_img.get_fdata()
    nib.save(new_img, nifti_path)
    return new_img_np

def combine_segmentations(tc, wt, et, out_path):
    segmentation1 = nib.load(tc).get_fdata()
    segmentation2 = nib.load(wt).get_fdata()
    segmentation3 = nib.load(et).get_fdata()
    combined_segmentation = np.stack((segmentation1, segmentation2, segmentation3), axis=-1)
    background_mask = (np.sum(combined_segmentation, axis=-1) == 0).astype(np.float32)
    combined_segmentation_with_background = np.concatenate((background_mask[..., np.newaxis], combined_segmentation), axis=-1)
    combined_nifti = nib.Nifti1Image(combined_segmentation_with_background, affine=nib.load(tc).affine)
    nib.save(combined_nifti, out_path)
    print(f'Combined segmentation saved to {out_path}')

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
        combined_path = os.path.join('./results', name)
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
        pred_tc = pred_mask[0][0]
        pred_wt = pred_mask[0][1]
        pred_et = pred_mask[0][2]
        combined_file_path = os.path.join(combined_path, f'{name}_TC.nii')
        tc_file_path = os.path.join(TC_path, f'{name}_TC.nii')
        wt_file_path = os.path.join(WT_path, f'{name}_WT.nii')
        et_file_path = os.path.join(ET_path, f'{name}_ET.nii')
        _ = resize(affine, pred_tc, tc_file_path)
        _ = resize(affine, pred_wt, wt_file_path)
        _ = resize(affine, pred_et, et_file_path)
        combine_segmentations(tc_file_path, wt_file_path, et_file_path,combined_file_path)
        # print("-------")
        # print(affine)
        # print("img")
        # print(img.shape)
        # print("mask")
        # print(pred_mask[0][0].shape)
        # print("-------")

        # pred_tc, affine = reorient_to_match(nib.Nifti1Image(pred_mask[0][0], affine), img)
        # pred_wt, affine = reorient_to_match(nib.Nifti1Image(pred_mask[0][1], affine), img)
        # pred_et, affine = reorient_to_match(nib.Nifti1Image(pred_mask[0][2], affine), img)

        # Flip the masks horizontally
        # pred_tc = flip_across_all_axes(pred_mask[0][0])
        # pred_wt = flip_across_all_axes(pred_mask[0][1])
        # pred_et = flip_across_all_axes(pred_mask[0][2])

        # Convert SimpleITK image back to numpy array for saving with nibabel
        # Extract individual masks for TC, WT, and ET

        
        # Create NIfTI images with the extracted masks and affine
        # tc_nifti = nib.Nifti1Image(pred_tc, affine)
        # wt_nifti = nib.Nifti1Image(pred_wt, affine)
        # et_nifti = nib.Nifti1Image(pred_et, affine)
        
        # nib.save(tc_nifti, tc_file_path)
        # nib.save(wt_nifti, wt_file_path)
        # nib.save(et_nifti, et_file_path)

        # tc_nifti = nib.Nifti1Image(pred_mask[0][0], affine)
        # wt_nifti = nib.Nifti1Image(pred_mask[0][1], affine)
        # et_nifti = nib.Nifti1Image(pred_mask[0][2], affine)
        # nib.save(tc_nifti, os.path.join(TC_path, f'{name}_TC.nii.gz'))
        # nib.save(wt_nifti, os.path.join(WT_path, f'{name}_WT.nii.gz'))
        # nib.save(et_nifti, os.path.join(ET_path, f'{name}_ET.nii.gz'))
        # TC_seg_path = os.path.join(TC_path, f'{name}_TC.nii')
        # WT_seg_path = os.path.join(WT_path, f'{name}_WT.nii')
        # ET_seg_path = os.path.join(ET_path, f'{name}_ET.nii')
        return tc_file_path, wt_file_path, et_file_path, combined_file_path
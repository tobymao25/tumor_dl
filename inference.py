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

# def reorient_to_match(img, target_img):
#     orig_ornt = axcodes2ornt(aff2axcodes(img.affine))
#     target_ornt = axcodes2ornt(aff2axcodes(target_img.affine))
#     transform_ornt = ornt_transform(orig_ornt, target_ornt)
#     return apply_orientation(img.get_fdata(), transform_ornt), target_img.affine

# def flip_across_all_axes(arr):
#     return np.flip(np.flip(np.flip(arr, axis=0), axis=1), axis=2)
def load_and_modify_image(image_path):
    img = nib.load(image_path)
    img_data = img.get_fdata()
    img_data = img_data[..., 0] 
    img_sitk = sitk.GetImageFromArray(img_data)
    img_sitk.SetOrigin(img.affine[:3, 3].tolist())
    img_sitk.SetSpacing((img.header['pixdim'][1], img.header['pixdim'][2], img.header['pixdim'][3]))
    img_sitk.SetDirection(np.reshape(img.affine[:3, :3], -1).tolist())
    return img_sitk, img.affine

def resize(img, pm):
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing(original_spacing)
    resampler.SetSize(original_size)
    resampled_mask = resampler.Execute(pm)
    return resampled_mask

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
        img, affine = load_and_modify_image(path)
        new_mask = resize(img, pred_mask)

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
        new_mask_array = sitk.GetArrayFromImage(new_mask)
    
        # Extract individual masks for TC, WT, and ET
        pred_tc = new_mask_array[0, :, :, :]
        pred_wt = new_mask_array[1, :, :, :]
        pred_et = new_mask_array[2, :, :, :]
        
        # Create NIfTI images with the extracted masks and affine
        tc_nifti = nib.Nifti1Image(pred_tc, affine)
        wt_nifti = nib.Nifti1Image(pred_wt, affine)
        et_nifti = nib.Nifti1Image(pred_et, affine)
        
        # Define file paths
        tc_file_path = os.path.join(TC_path, f'{name}_TC.nii')
        wt_file_path = os.path.join(WT_path, f'{name}_WT.nii')
        et_file_path = os.path.join(ET_path, f'{name}_ET.nii')

        nib.save(tc_nifti, tc_file_path)
        nib.save(wt_nifti, wt_file_path)
        nib.save(et_nifti, et_file_path)

        # tc_nifti = nib.Nifti1Image(pred_mask[0][0], affine)
        # wt_nifti = nib.Nifti1Image(pred_mask[0][1], affine)
        # et_nifti = nib.Nifti1Image(pred_mask[0][2], affine)
        # nib.save(tc_nifti, os.path.join(TC_path, f'{name}_TC.nii.gz'))
        # nib.save(wt_nifti, os.path.join(WT_path, f'{name}_WT.nii.gz'))
        # nib.save(et_nifti, os.path.join(ET_path, f'{name}_ET.nii.gz'))
        TC_seg_path = os.path.join(TC_path, f'{name}_TC.nii')
        WT_seg_path = os.path.join(WT_path, f'{name}_WT.nii')
        ET_seg_path = os.path.join(ET_path, f'{name}_ET.nii')
        return TC_seg_path, WT_seg_path, ET_seg_path
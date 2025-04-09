import os
from PIL import Image
from sklearn.decomposition import PCA
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from lidar_cam_calib import LidarCamCalib
import cv2
from copy import deepcopy
from dino_vit_extractor import ViTExtractor
from clipdino_utils import run_pca, viz_pca3, resize_to_match_aspect_ratio, compute_new_dims


class DinoV2:
    def __init__(self, model_name: str = "dinov2_vitl14"):
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.get_dinov2_model(model_name)

    def get_transform(self, smaller_side_size: int) -> transforms.Compose:
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        interpolation_mode = transforms.InterpolationMode.LANCZOS
        transform = transforms.Compose([
            transforms.Resize(size=smaller_side_size, interpolation=interpolation_mode, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
        return transform

    @torch.inference_mode()
    def get_dinov2_model(self, model_name: str = "dinov2_vitl14"):
        model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=model_name)
        model.eval()
        model.to(self.DEVICE)
        return model

    @torch.inference_mode()
    def get_patch_features(self, pil_img: Image.Image):
        transform = self.get_transform(min(pil_img.size))
        image_tensor = transform(pil_img).to(self.DEVICE)
        height, width = image_tensor.shape[1:]  # C x H x W
        assert height == pil_img.size[1] and width == pil_img.size[0], "Debug: Image size changed after transform"
        cropped_width, cropped_height = width - width % self.model.patch_size, height - height % self.model.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]
        grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)  # h x w
        image_batch = image_tensor.unsqueeze(0)
        tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()  # (num_patches, dim) = (h*w, dim)
        return tokens.cpu(), grid_size, cropped_height, cropped_width


if __name__ == "__main__":
    with torch.inference_mode():
        lcc = LidarCamCalib(ros_flag=False, robotname="spot", cam_res=3072)
        cv2_img = cv2.imread("/home/dynamo/AMRL_Research/repos/robot_calib/notrack_bags/paths/poc_cut/images/1742591008774347085.png")
        cv2_img = lcc.cam_calib.rectifyRawCamImage(cv2_img)
        pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)).convert("RGB")
        # pil_img = pil_img.resize((2048, 1536))

        # using your minimal class
        dino = DinoV2("dinov2_vitb14")
        tokens, grid_size, cropped_h, cropped_w = dino.get_patch_features(pil_img)
        projected_tokens = run_pca(tokens, n_components=3)
        img = viz_pca3(projected_tokens, grid_size, cropped_w, cropped_h)
        plt.imshow(img)
        plt.show()

        # using general class (from Distilled Feature Fields paper)
        # to make it 14 (patch size) divisible, without changing aspect ratio by much; needed when using DFF's parameters, to work properly with dinov2
        # 224 is the input resolution for this visual encoder
        # pil_img = resize_to_match_aspect_ratio(pil_img, target_ratio_tuple=compute_new_dims(orig_size=pil_img.size, short_side=224, round_multiple=14))
        extractor = ViTExtractor(model_type='dinov2_vitb14', stride=None)  # DFF uses stride=4, and dino_vits8
        image_batch, _, cropped_h, cropped_w = extractor.preprocess(image_path=pil_img,
                                                                    load_size=-1,  # DFF uses load_size 224
                                                                    allow_crop=True)   # DFF uses allow_crop=False
        descriptors = extractor.extract_descriptors(batch=image_batch,
                                                    layer=len(extractor.model.blocks) - 1,  # last layer
                                                    facet='token',  # DFF uses 'key'
                                                    bin=False,
                                                    include_cls=False)
        descriptors = descriptors.cpu().squeeze()
        projected_tokens = run_pca(descriptors, n_components=3)
        img = viz_pca3(projected_tokens, extractor.num_patches, cropped_w, cropped_h)
        plt.imshow(img)
        plt.show()

        # img.save("pca_img.png")

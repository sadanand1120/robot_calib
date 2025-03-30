import os
# os.environ["XFORMERS_DISABLED"] = "1" # Switch to enable xFormers
from PIL import Image
from sklearn.decomposition import PCA
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from lidar_cam_calib import LidarCamCalib
import cv2


class DinoV2:
    def __init__(self, model_name: str = "dinov2_vitl14"):
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.get_dinov2_model(model_name)

    def get_transform(self, smaller_side_size: int) -> transforms.Compose:
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        interpolation_mode = transforms.InterpolationMode.BICUBIC
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
        cropped_width, cropped_height = width - width % self.model.patch_size, height - height % self.model.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]
        grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)  # h x w
        image_batch = image_tensor.unsqueeze(0)
        tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()  # (num_patches, dim) = (h*w, dim)
        return tokens.cpu(), grid_size

    @torch.inference_mode()
    def run_pca(self, tokens, n_components):
        pca = PCA(n_components=n_components)
        pca.fit(tokens)
        projected_tokens = pca.transform(tokens)
        return projected_tokens

    @torch.inference_mode()
    def viz_pca3(self, projected_tokens, grid_size, orig_img_width, orig_img_height, resample=Image.NEAREST) -> Image:
        t = torch.tensor(projected_tokens)
        t_min = t.min(dim=0, keepdim=True).values
        t_max = t.max(dim=0, keepdim=True).values
        normalized_t = (t - t_min) / (t_max - t_min)
        array = (normalized_t * 255).byte().numpy()
        array = array.reshape(*grid_size, 3)
        return Image.fromarray(array).resize((orig_img_width, orig_img_height), resample=resample)


if __name__ == "__main__":
    dino = DinoV2("dinov2_vitl14")
    lcc = LidarCamCalib(ros_flag=False, robotname="spot", cam_res=3072)
    cv2_img = cv2.imread("/home/dynamo/AMRL_Research/repos/robot_calib/notrack_bags/paths/poc_cut/images/1742591207921454886.png")
    cv2_img = lcc.cam_calib.rectifyRawCamImage(cv2_img)
    pil_img = Image.fromarray(cv2_img)
    pil_img = pil_img.resize((2048, 1536))
    tokens, grid_size = dino.get_patch_features(pil_img)
    projected_tokens = dino.run_pca(tokens, n_components=3)
    img = dino.viz_pca3(projected_tokens, grid_size, *pil_img.size)
    plt.imshow(img)
    plt.show()

    # img.save("pca_img.png")

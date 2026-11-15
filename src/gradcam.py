"""
Module: src.gradcam
Gradient-weighted Class Activation Mapping (Grad-CAM) implementation for
interpreting Deep Learning decisions on Retinal OCT images.
Highlights pathological biomarkers such as choroidal neovascularization,
macular fluid cysts (DME), and drusen deposits.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm


class GradCAM:
    """
    Grad-CAM class for visualizing convolutional layer activations.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        # Disable inplace activations to prevent backward hook view modification errors
        for m in self.model.modules():
            if hasattr(m, "inplace"):
                m.inplace = False
        self.target_layer = target_layer or self._find_default_target_layer()
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _find_default_target_layer(self):
        # Automatically detect the last convolutional layer
        if hasattr(self.model, "layer4"):
            # ResNet
            return self.model.layer4[-1]
        elif hasattr(self.model, "features"):
            # DenseNet, MobileNet, VGG, or CustomOCTCNN
            for module in reversed(list(self.model.features.modules())):
                if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                    return module
            return self.model.features[-1]
        else:
            # Fallback: scan all submodules for Conv2d
            last_conv = None
            for module in self.model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = module
            if last_conv is not None:
                return last_conv
            raise ValueError("Could not automatically locate a convolutional layer for Grad-CAM.")

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach().clone()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach().clone()

        self.f_hook = self.target_layer.register_forward_hook(forward_hook)
        self.b_hook = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        if hasattr(self, "f_hook") and self.f_hook is not None:
            self.f_hook.remove()
        if hasattr(self, "b_hook") and self.b_hook is not None:
            self.b_hook.remove()

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generates Grad-CAM heatmap for a single input tensor (1, C, H, W).
        Returns:
            heatmap (numpy.ndarray, shape H x W, float in [0, 1]),
            pred_class (int),
            confidence (float),
            all_probs (numpy.ndarray)
        """
        for param in self.model.parameters():
            param.requires_grad_(True)
        self.model.zero_grad()
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1).detach().cpu().numpy()[0]
        
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])
        
        if target_class is None:
            target_class = pred_class

        # Backward pass for the target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Global average pooling of gradients
        # gradients: [1, C, H_feat, W_feat]
        # activations: [1, C, H_feat, W_feat]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            heatmap = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            heatmap = np.zeros_like(cam_np)

        return heatmap, pred_class, confidence, probs

    @staticmethod
    def overlay_heatmap_on_image(original_image, heatmap, alpha=0.5, colormap="jet"):
        """
        Blends heatmap with original PIL or numpy image.
        Returns blended PIL Image.
        """
        if isinstance(original_image, Image.Image):
            pil_img = original_image.convert("RGB")
            w, h = pil_img.size
        else:
            pil_img = Image.fromarray(original_image).convert("RGB")
            w, h = pil_img.size

        # Colorize heatmap
        import matplotlib as mpl
        try:
            cmap = mpl.colormaps[colormap]
        except AttributeError:
            cmap = cm.get_cmap(colormap)
        colored_heatmap = cmap(heatmap)[:, :, :3]  # drop alpha, shape (H, W, 3) in [0, 1]
        colored_heatmap_uint8 = (colored_heatmap * 255).astype(np.uint8)
        heatmap_img = Image.fromarray(colored_heatmap_uint8).resize((w, h), Image.Resampling.BILINEAR)

        blended = Image.blend(pil_img, heatmap_img, alpha=alpha)
        return blended, heatmap_img

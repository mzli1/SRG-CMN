import torch
import torch.nn as nn
import torchvision.models as models
from .sagnn import OptimizedSAGNNAdapter

class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        
        self.sagnn_adapter = OptimizedSAGNNAdapter(
            in_dim=2048,
            hidden_dim=args.d_model, 
            out_dim=2048,            
            num_heads=4
        )

    def forward(self, images):
        patch_feats_map = self.model(images)
        # 获取融合后的 patch_feats
        patch_feats, global_z = self.sagnn_adapter(patch_feats_map)
        return patch_feats, global_z
import torch
import torch.nn as nn

class SpatialGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1): 
        super(SpatialGATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        
        self.linear_proj = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_heads, out_dim))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_heads, out_dim))
        
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

    def forward(self, h, adj):
        B, N, _ = h.size()
        h_proj = self.linear_proj(h).view(B, N, self.num_heads, self.out_dim)
        h_proj = self.dropout(h_proj)
        
        scores_source = (h_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (h_proj * self.scoring_fn_target).sum(dim=-1)
        scores = scores_source.unsqueeze(2) + scores_target.unsqueeze(1)
        scores = self.leakyReLU(scores)
        
        if adj is not None:
            scores = scores.masked_fill(adj.unsqueeze(-1) == 0, -1e9)
        
        attn_coef = self.softmax(scores)
        attn_coef = self.dropout(attn_coef)
        
        out = torch.einsum('bijh,bjhd->bihd', attn_coef, h_proj)
        out = out.reshape(B, N, self.num_heads * self.out_dim)
        return out

class OptimizedSAGNNAdapter(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048, num_heads=4):
        super(OptimizedSAGNNAdapter, self).__init__()
        
        # 1. 降维
        self.proj_in = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 2. 空间 GAT
        self.gat_layer = SpatialGATLayer(
            in_dim=hidden_dim,
            out_dim=hidden_dim // num_heads,
            num_heads=num_heads,
            dropout=0.1
        )
        
        # 3. 升维 (GNN分支的输出)
        self.proj_back = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            # 对 GNN 的输出做 LN ，为了稳定 GNN 自身的数值
            nn.LayerNorm(out_dim), 
            nn.Dropout(0.1)
        )
        
        # 4. 全局聚合
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pool_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.proj_global = nn.Linear(hidden_dim, out_dim)

        # 使用 Tanh：允许 GNN 对原始特征做加法或减法修正
        self.gate = nn.Parameter(torch.tensor([0.54]))
        
        self.gate_global = nn.Parameter(torch.zeros(1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    八邻域取得最佳性能
    def _build_grid_graph(self, H, W, device):
        num_nodes = H * W
        adj = torch.eye(num_nodes, device=device)
        for r in range(H):
            for c in range(W):
                idx = r * W + c
                if c + 1 < W: adj[idx, idx+1] = adj[idx+1, idx] = 1
                if r + 1 < H: adj[idx, idx+W] = adj[idx+W, idx] = 1
                if r + 1 < H and c + 1 < W: 
                    idx2 = (r+1)*W + (c+1); adj[idx, idx2] = adj[idx2, idx] = 1
                if r + 1 < H and c - 1 >= 0:
                    idx2 = (r+1)*W + (c-1); adj[idx, idx2] = adj[idx2, idx] = 1
        return adj

    # def _build_grid_graph(self, H, W, device):
    #     """消融实验变体：4-邻域 (曼哈顿距离)"""
    #     num_nodes = H * W
    #     adj = torch.eye(num_nodes, device=device)
    #     for r in range(H):
    #         for c in range(W):
    #             idx = r * W + c
                
    #             # 1. 向右 (Right)
    #             if c + 1 < W:
    #                 adj[idx, idx+1] = adj[idx+1, idx] = 1
                
    #             # 2. 向下 (Down)
    #             if r + 1 < H:
    #                 adj[idx, idx+W] = adj[idx+W, idx] = 1
                
    #             # 这里删除了对角线连接，即删除了8-邻域的特征
    #     return adj

    def forward(self, x_map):
        B, C, H, W = x_map.size()
        N = H * W
        
        # 1. 原始特征：保持不动
        patch_feats_raw = x_map.view(B, C, N).permute(0, 2, 1) # [B, N, 2048]
        
        # 2. GNN 分支计算
        x_embed = self.proj_in(patch_feats_raw) 
        adj = self._build_grid_graph(H, W, x_map.device).unsqueeze(0).expand(B, -1, -1)
        x_struct = self.gat_layer(x_embed, adj) 
        
        # 得到 GNN 的修正量
        patch_delta = self.proj_back(x_struct) # [B, N, 2048]
        # 随机噪声测试
        # patch_delta = torch.randn_like(patch_feats_raw)
        
        # 3. 残差融合
        # Final = Raw + Gate * GNN
        # 这样在 gate=0 时，模型完全等同于 Baseline
        patch_feats_final = patch_feats_raw + torch.tanh(self.gate) * patch_delta
        
        # Global 分支 (保持原有逻辑)
        query = self.pool_query.expand(B, -1, -1)
        z_struct, _ = self.pool_attn(query, x_struct, x_struct)
        z_struct = z_struct.squeeze(1)
        z_sagnn = self.proj_global(z_struct)
        z_resnet_avg = self.avg_pool(x_map).view(B, C)
        z_final = z_resnet_avg + torch.tanh(self.gate_global) * z_sagnn
        
        return patch_feats_final, z_final
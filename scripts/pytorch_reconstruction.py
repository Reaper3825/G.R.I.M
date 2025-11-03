"""
PyTorch Model Reconstruction for llama3.2-vision
Step 2: Build PyTorch vision encoder from extracted weights
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Dict

# Model architecture from metadata
EMBED_DIM = 1280
NUM_HEADS = 16
FFN_DIM = 5120
NUM_BLOCKS = 32
NUM_GLOBAL_BLOCKS = 8
IMAGE_SIZE = 560
PATCH_SIZE = 14
NUM_CHANNELS = 3

class VisionAttention(nn.Module):
    """Multi-head self-attention for vision encoder"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
    def forward(self, x):
        B, N, C = x.shape
        
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        return out

class VisionFFN(nn.Module):
    """Feed-forward network for vision encoder"""
    def __init__(self, embed_dim, ffn_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim, bias=True)
        self.fc2 = nn.Linear(ffn_dim, embed_dim, bias=True)
        self.activation = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

class VisionEncoderBlock(nn.Module):
    """Single transformer block for vision encoder"""
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = VisionAttention(embed_dim, num_heads)
        
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = VisionFFN(embed_dim, ffn_dim)
        
    def forward(self, x):
        # Attention with residual
        x = x + self.attn(self.attn_norm(x))
        
        # FFN with residual
        x = x + self.ffn(self.ffn_norm(x))
        
        return x

class GlobalVisionEncoderBlock(nn.Module):
    """Global transformer block with gating"""
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super().__init__()
        
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = VisionAttention(embed_dim, num_heads)
        self.attn_gate = nn.Parameter(torch.ones(1))
        
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = VisionFFN(embed_dim, ffn_dim)
        self.ffn_gate = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # Gated attention with residual
        x = x + self.attn_gate * self.attn(self.attn_norm(x))
        
        # Gated FFN with residual
        x = x + self.ffn_gate * self.ffn(self.ffn_norm(x))
        
        return x

class MllamaVisionEncoder(nn.Module):
    """Complete Mllama vision encoder"""
    def __init__(self):
        super().__init__()
        
        # Patch embedding (convolutional)
        self.patch_embed = nn.Conv2d(
            NUM_CHANNELS, 
            EMBED_DIM, 
            kernel_size=PATCH_SIZE, 
            stride=PATCH_SIZE,
            bias=False
        )
        
        # Pre-normalization
        self.pre_ln = nn.LayerNorm(EMBED_DIM)
        
        # Tile position embeddings (for multi-tile images) - 5120 dim (4x embed_dim)
        self.pre_tile_pos_embed = nn.Parameter(torch.zeros(5120, 9))
        self.pre_tile_pos_gate = nn.Parameter(torch.ones(1))
        self.post_tile_pos_embed = nn.Parameter(torch.zeros(5120, 9))
        self.post_tile_pos_gate = nn.Parameter(torch.ones(1))
        
        # Local vision encoder blocks (32 blocks)
        self.blocks = nn.ModuleList([
            VisionEncoderBlock(EMBED_DIM, NUM_HEADS, FFN_DIM)
            for _ in range(NUM_BLOCKS)
        ])
        
        # Global vision encoder blocks (8 blocks)
        self.global_blocks = nn.ModuleList([
            GlobalVisionEncoderBlock(EMBED_DIM, NUM_HEADS, FFN_DIM)
            for _ in range(NUM_GLOBAL_BLOCKS)
        ])
        
        # Post-normalization
        self.post_ln = nn.LayerNorm(EMBED_DIM)
        
    def forward(self, pixel_values):
        # Patch embedding
        x = self.patch_embed(pixel_values)  # (B, C, H, W) -> (B, embed_dim, H/patch_size, W/patch_size)
        
        # Flatten and transpose
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C) where N = H*W
        
        # Pre-normalization
        x = self.pre_ln(x)
        
        # Local encoder blocks
        for block in self.blocks:
            x = block(x)
        
        # Global encoder blocks
        for block in self.global_blocks:
            x = block(x)
        
        # Post-normalization
        x = self.post_ln(x)
        
        return x

def load_weights_into_model(model: MllamaVisionEncoder, tensors: Dict[str, np.ndarray]):
    """Load extracted GGUF weights into PyTorch model"""
    print("[PyTorch] Loading weights into model...")
    
    state_dict = {}
    loaded_count = 0
    
    # Patch embedding
    if 'v.patch_embd.weight' in tensors:
        # GGUF format: (kernel_h, kernel_w, in_channels, out_channels)
        # PyTorch format: (out_channels, in_channels, kernel_h, kernel_w)
        patch_weight = tensors['v.patch_embd.weight']
        patch_weight = np.transpose(patch_weight, (3, 2, 0, 1))
        state_dict['patch_embed.weight'] = torch.from_numpy(patch_weight)
        loaded_count += 1
        print(f"  ✅ Loaded patch_embed.weight: {patch_weight.shape}")
    
    # Pre/Post LayerNorm
    for src, dst in [
        ('v.pre_ln.weight', 'pre_ln.weight'),
        ('v.pre_ln.bias', 'pre_ln.bias'),
        ('v.post_ln.weight', 'post_ln.weight'),
        ('v.post_ln.bias', 'post_ln.bias'),
    ]:
        if src in tensors:
            state_dict[dst] = torch.from_numpy(tensors[src])
            loaded_count += 1
    
    # Tile position embeddings
    for src, dst in [
        ('v.pre_tile_position_embd.weight', 'pre_tile_pos_embed'),
        ('v.pre_tile_position_embd.gate', 'pre_tile_pos_gate'),
        ('v.post_tile_position_embd.weight', 'post_tile_pos_embed'),
        ('v.post_tile_position_embd.gate', 'post_tile_pos_gate'),
    ]:
        if src in tensors:
            state_dict[dst] = torch.from_numpy(tensors[src])
            loaded_count += 1
    
    # Local blocks
    for i in range(NUM_BLOCKS):
        prefix = f'v.blk.{i}'
        
        # Attention weights
        for weight_type in ['attn_q', 'attn_k', 'attn_v', 'attn_output']:
            src = f'{prefix}.{weight_type}.weight'
            if src in tensors:
                # Transpose from (in_features, out_features) to (out_features, in_features)
                weight = tensors[src].T
                dst = f'blocks.{i}.attn.{weight_type.replace("attn_", "")}_proj.weight'
                if weight_type == 'attn_output':
                    dst = f'blocks.{i}.attn.out_proj.weight'
                state_dict[dst] = torch.from_numpy(weight)
                loaded_count += 1
        
        # Attention LayerNorm
        for param in ['weight', 'bias']:
            src = f'{prefix}.attn_norm.{param}'
            if src in tensors:
                state_dict[f'blocks.{i}.attn_norm.{param}'] = torch.from_numpy(tensors[src])
                loaded_count += 1
        
        # FFN weights
        for weight_type in ['ffn_up', 'ffn_down']:
            src_w = f'{prefix}.{weight_type}.weight'
            src_b = f'{prefix}.{weight_type}.bias'
            
            if src_w in tensors:
                weight = tensors[src_w].T
                fc_name = 'fc1' if weight_type == 'ffn_up' else 'fc2'
                state_dict[f'blocks.{i}.ffn.{fc_name}.weight'] = torch.from_numpy(weight)
                loaded_count += 1
            
            if src_b in tensors:
                fc_name = 'fc1' if weight_type == 'ffn_up' else 'fc2'
                state_dict[f'blocks.{i}.ffn.{fc_name}.bias'] = torch.from_numpy(tensors[src_b])
                loaded_count += 1
        
        # FFN LayerNorm
        for param in ['weight', 'bias']:
            src = f'{prefix}.ffn_norm.{param}'
            if src in tensors:
                state_dict[f'blocks.{i}.ffn_norm.{param}'] = torch.from_numpy(tensors[src])
                loaded_count += 1
    
    # Global blocks
    for i in range(NUM_GLOBAL_BLOCKS):
        prefix = f'v.global.blk.{i}'
        
        # Attention weights
        for weight_type in ['attn_q', 'attn_k', 'attn_v', 'attn_output']:
            src = f'{prefix}.{weight_type}.weight'
            if src in tensors:
                weight = tensors[src].T
                dst = f'global_blocks.{i}.attn.{weight_type.replace("attn_", "")}_proj.weight'
                if weight_type == 'attn_output':
                    dst = f'global_blocks.{i}.attn.out_proj.weight'
                state_dict[dst] = torch.from_numpy(weight)
                loaded_count += 1
        
        # Gates
        for gate_type in ['attn_gate', 'ffn_gate']:
            src = f'{prefix}.{gate_type}'
            if src in tensors:
                state_dict[f'global_blocks.{i}.{gate_type}'] = torch.from_numpy(tensors[src])
                loaded_count += 1
        
        # Attention LayerNorm
        for param in ['weight', 'bias']:
            src = f'{prefix}.attn_norm.{param}'
            if src in tensors:
                state_dict[f'global_blocks.{i}.attn_norm.{param}'] = torch.from_numpy(tensors[src])
                loaded_count += 1
        
        # FFN weights
        for weight_type in ['ffn_up', 'ffn_down']:
            src_w = f'{prefix}.{weight_type}.weight'
            src_b = f'{prefix}.{weight_type}.bias'
            
            if src_w in tensors:
                weight = tensors[src_w].T
                fc_name = 'fc1' if weight_type == 'ffn_up' else 'fc2'
                state_dict[f'global_blocks.{i}.ffn.{fc_name}.weight'] = torch.from_numpy(weight)
                loaded_count += 1
            
            if src_b in tensors:
                fc_name = 'fc1' if weight_type == 'ffn_up' else 'fc2'
                state_dict[f'global_blocks.{i}.ffn.{fc_name}.bias'] = torch.from_numpy(tensors[src_b])
                loaded_count += 1
        
        # FFN LayerNorm
        for param in ['weight', 'bias']:
            src = f'{prefix}.ffn_norm.{param}'
            if src in tensors:
                state_dict[f'global_blocks.{i}.ffn_norm.{param}'] = torch.from_numpy(tensors[src])
                loaded_count += 1
    
    print(f"\n  ✅ Loaded {loaded_count} parameters into PyTorch model")
    
    # Load into model
    model.load_state_dict(state_dict, strict=False)
    
    return model

def main():
    print("="*80)
    print("PyTorch Model Reconstruction - Step 2")
    print("="*80)
    
    # Load extracted tensors
    tensors_path = Path("D:/G.R.I.M/data/models/vision/llama3.2-vision-extracted/vision_tensors.npz")
    metadata_path = Path("D:/G.R.I.M/data/models/vision/llama3.2-vision-extracted/metadata.json")
    
    if not tensors_path.exists():
        print(f"❌ Tensors file not found: {tensors_path}")
        return
    
    print(f"[1/4] Loading extracted tensors...")
    tensors_data = np.load(tensors_path)
    tensors = {k: tensors_data[k] for k in tensors_data.files}
    print(f"  ✅ Loaded {len(tensors)} tensors")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"  ✅ Loaded metadata")
    
    # Create model
    print(f"\n[2/4] Creating PyTorch model...")
    model = MllamaVisionEncoder()
    print(f"  ✅ Model created with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    # Load weights
    print(f"\n[3/4] Loading weights...")
    model = load_weights_into_model(model, tensors)
    
    # Set to eval mode
    model.eval()
    
    # Save PyTorch model
    output_dir = Path("D:/G.R.I.M/data/models/vision/llama3.2-vision-pytorch")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[4/4] Saving PyTorch model...")
    torch.save(model.state_dict(), output_dir / "vision_encoder.pth")
    print(f"  ✅ Saved to: {output_dir / 'vision_encoder.pth'}")
    
    # Save full model for ONNX export
    torch.save(model, output_dir / "vision_encoder_full.pth")
    print(f"  ✅ Saved full model to: {output_dir / 'vision_encoder_full.pth'}")
    
    print("\n" + "="*80)
    print("✅ Step 2 Complete!")
    print("="*80)
    print(f"PyTorch model ready for ONNX export")
    print(f"Next: Run onnx_export.py to convert to ONNX")

if __name__ == "__main__":
    main()

"""
GGUF to PyTorch Converter for llama3.2-vision
Step 1: Parse GGUF format and extract tensors
"""

import sys
import io
# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

# GGUF Format Constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

# GGUF value types
class GGUFValueType:
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12

# GGML tensor types (quantization formats)
class GGMLType:
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15

class GGUFReader:
    def __init__(self, path: Path):
        self.path = path
        self.file = open(path, 'rb')
        self.metadata = {}
        self.tensors = {}
        
    def read_u32(self):
        return struct.unpack('<I', self.file.read(4))[0]
    
    def read_u64(self):
        return struct.unpack('<Q', self.file.read(8))[0]
    
    def read_i32(self):
        return struct.unpack('<i', self.file.read(4))[0]
    
    def read_f32(self):
        return struct.unpack('<f', self.file.read(4))[0]
    
    def read_string(self):
        length = self.read_u64()
        return self.file.read(length).decode('utf-8')
    
    def read_value(self, value_type):
        """Read a value based on its type"""
        if value_type == GGUFValueType.UINT8:
            return struct.unpack('B', self.file.read(1))[0]
        elif value_type == GGUFValueType.INT8:
            return struct.unpack('b', self.file.read(1))[0]
        elif value_type == GGUFValueType.UINT16:
            return struct.unpack('<H', self.file.read(2))[0]
        elif value_type == GGUFValueType.INT16:
            return struct.unpack('<h', self.file.read(2))[0]
        elif value_type == GGUFValueType.UINT32:
            return self.read_u32()
        elif value_type == GGUFValueType.INT32:
            return self.read_i32()
        elif value_type == GGUFValueType.FLOAT32:
            return self.read_f32()
        elif value_type == GGUFValueType.UINT64:
            return self.read_u64()
        elif value_type == GGUFValueType.INT64:
            return struct.unpack('<q', self.file.read(8))[0]
        elif value_type == GGUFValueType.FLOAT64:
            return struct.unpack('<d', self.file.read(8))[0]
        elif value_type == GGUFValueType.BOOL:
            return struct.unpack('?', self.file.read(1))[0]
        elif value_type == GGUFValueType.STRING:
            return self.read_string()
        elif value_type == GGUFValueType.ARRAY:
            array_type = self.read_u32()
            array_len = self.read_u64()
            return [self.read_value(array_type) for _ in range(array_len)]
        else:
            raise ValueError(f"Unknown value type: {value_type}")
    
    def parse_header(self):
        """Parse GGUF header"""
        print("[GGUF] Parsing header...")
        
        # Read magic number
        magic = self.read_u32()
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {hex(magic)}")
        print(f"  ✅ Magic: {hex(magic)}")
        
        # Read version
        version = self.read_u32()
        print(f"  ✅ Version: {version}")
        
        # Read tensor count
        tensor_count = self.read_u64()
        print(f"  ✅ Tensor count: {tensor_count}")
        
        # Read metadata count
        metadata_count = self.read_u64()
        print(f"  ✅ Metadata count: {metadata_count}")
        
        return tensor_count, metadata_count
    
    def parse_metadata(self, count):
        """Parse metadata key-value pairs"""
        print(f"\n[GGUF] Parsing {count} metadata entries...")
        
        for i in range(count):
            key = self.read_string()
            value_type = self.read_u32()
            value = self.read_value(value_type)
            
            self.metadata[key] = value
            
            # Print important metadata
            if i < 20 or 'vision' in key.lower() or 'clip' in key.lower():
                value_str = str(value)[:100]
                print(f"  [{i}] {key} = {value_str}")
        
        print(f"\n  Total metadata keys: {len(self.metadata)}")
        
        # Look for vision-specific metadata
        vision_keys = [k for k in self.metadata.keys() if 'vision' in k.lower() or 'clip' in k.lower() or 'image' in k.lower()]
        if vision_keys:
            print(f"  Vision-related keys found: {vision_keys}")
    
    def parse_tensor_info(self, count):
        """Parse tensor information (names, shapes, types)"""
        print(f"\n[GGUF] Parsing {count} tensor info entries...")
        
        tensor_info = []
        
        for i in range(count):
            name = self.read_string()
            n_dims = self.read_u32()
            
            shape = []
            for _ in range(n_dims):
                shape.append(self.read_u64())
            
            dtype = self.read_u32()
            offset = self.read_u64()
            
            tensor_info.append({
                'name': name,
                'shape': shape,
                'dtype': dtype,
                'offset': offset
            })
            
            # Print vision encoder tensors
            if 'vision' in name.lower() or 'clip' in name.lower() or 'image' in name.lower():
                dtype_name = self.get_dtype_name(dtype)
                print(f"  [{i}] {name}")
                print(f"      Shape: {shape}, Type: {dtype_name}, Offset: {offset}")
        
        return tensor_info
    
    def get_dtype_name(self, dtype):
        """Get human-readable dtype name"""
        dtype_map = {
            GGMLType.F32: "F32",
            GGMLType.F16: "F16",
            GGMLType.Q4_0: "Q4_0",
            GGMLType.Q4_1: "Q4_1",
            GGMLType.Q5_0: "Q5_0",
            GGMLType.Q5_1: "Q5_1",
            GGMLType.Q8_0: "Q8_0",
            GGMLType.Q2_K: "Q2_K",
            GGMLType.Q3_K: "Q3_K",
            GGMLType.Q4_K: "Q4_K",
            GGMLType.Q5_K: "Q5_K",
            GGMLType.Q6_K: "Q6_K",
        }
        return dtype_map.get(dtype, f"Unknown({dtype})")
    
    def dequantize_q4_0(self, data, shape):
        """Dequantize Q4_0 format to FP32"""
        # Q4_0: 4-bit quantization with 32-element blocks
        # Each block: 2 bytes (FP16 scale) + 16 bytes (32 x 4-bit values)
        block_size = 32
        total_elements = np.prod(shape)
        num_blocks = (total_elements + block_size - 1) // block_size
        
        result = np.zeros(total_elements, dtype=np.float32)
        
        offset = 0
        for block_idx in range(num_blocks):
            # Read scale (FP16)
            scale_bytes = data[offset:offset+2]
            scale = np.frombuffer(scale_bytes, dtype=np.float16)[0]
            offset += 2
            
            # Read 32 4-bit values (16 bytes)
            quant_bytes = data[offset:offset+16]
            offset += 16
            
            # Unpack 4-bit values
            for i in range(min(block_size, total_elements - block_idx * block_size)):
                byte_idx = i // 2
                if i % 2 == 0:
                    val = quant_bytes[byte_idx] & 0x0F
                else:
                    val = (quant_bytes[byte_idx] >> 4) & 0x0F
                
                # Convert to signed (-8 to 7)
                if val > 7:
                    val -= 16
                
                # Dequantize
                result[block_idx * block_size + i] = val * float(scale)
        
        return result.reshape(shape)
    
    def dequantize_q4_k(self, data, shape):
        """Dequantize Q4_K format to FP32"""
        # Q4_K: Advanced 4-bit quantization with super-blocks (256 elements)
        # More complex than Q4_0 - uses k-means clustering
        QK_K = 256
        K_SCALE_SIZE = 12
        
        total_elements = int(np.prod(shape))
        num_blocks = (total_elements + QK_K - 1) // QK_K
        result = np.zeros(total_elements, dtype=np.float32)
        
        offset = 0
        for block_idx in range(num_blocks):
            # Each Q4_K block structure:
            # - scales: 6x FP16 (12 bytes)
            # - min_scales: 6x FP16 (12 bytes)  
            # - quants: 128 bytes (256 x 4-bit)
            
            if offset + 144 > len(data):  # 12 + 12 + 128 = 152, but check safely
                break
            
            # Read scales (6 x FP16)
            scales = np.frombuffer(data[offset:offset+12], dtype=np.float16)
            offset += 12
            
            # Read min scales (6 x FP16)
            min_scales = np.frombuffer(data[offset:offset+12], dtype=np.float16)
            offset += 12
            
            # Read quantized data (128 bytes = 256 x 4-bit values)
            quant_bytes = data[offset:offset+128]
            offset += 128
            
            # Dequantize 256 values
            for i in range(min(QK_K, total_elements - block_idx * QK_K)):
                # Determine which sub-block (0-5) this element belongs to
                # 256 elements / 6 sub-blocks ≈ 42.67 elements per sub-block
                # But Q4_K uses groups of 32, so: 256/32 = 8 groups, mapped to 6 scales
                sub_block = min(i // 43, 5)  # 256/6 ≈ 43, clamp to 0-5
                
                # Get the 4-bit quantized value
                byte_idx = i // 2
                if i % 2 == 0:
                    q = quant_bytes[byte_idx] & 0x0F
                else:
                    q = (quant_bytes[byte_idx] >> 4) & 0x0F
                
                # Dequantize using scale and min
                scale = float(scales[sub_block])
                min_val = float(min_scales[sub_block])
                result[block_idx * QK_K + i] = q * scale + min_val
        
        return result.reshape(shape)
    
    def dequantize_q6_k(self, data, shape):
        """Dequantize Q6_K format to FP32"""
        # Q6_K: 6-bit quantization with super-blocks (256 elements)
        QK_K = 256
        
        total_elements = int(np.prod(shape))
        num_blocks = (total_elements + QK_K - 1) // QK_K
        result = np.zeros(total_elements, dtype=np.float32)
        
        offset = 0
        for block_idx in range(num_blocks):
            # Each Q6_K block structure:
            # - scales: 16x FP16 (32 bytes)
            # - quants: 192 bytes (256 x 6-bit)
            
            if offset + 224 > len(data):  # 32 + 192 = 224
                break
            
            # Read scales (16 x FP16) - each scale covers 16 elements
            scales = np.frombuffer(data[offset:offset+32], dtype=np.float16)
            offset += 32
            
            # Read quantized data (192 bytes)
            # 6 bits per value: 256 values * 6 bits = 1536 bits = 192 bytes
            quant_data = data[offset:offset+192]
            offset += 192
            
            # Dequantize 256 values
            bit_offset = 0
            for i in range(min(QK_K, total_elements - block_idx * QK_K)):
                # Extract 6-bit value
                byte_pos = bit_offset // 8
                bit_pos = bit_offset % 8
                
                if bit_pos <= 2:  # Fits in current + next byte
                    # Read 2 bytes and extract 6 bits
                    if byte_pos + 1 < len(quant_data):
                        two_bytes = (quant_data[byte_pos] | (quant_data[byte_pos + 1] << 8))
                        q = (two_bytes >> bit_pos) & 0x3F  # 0x3F = 63 (6 bits)
                    else:
                        q = (quant_data[byte_pos] >> bit_pos) & 0x3F
                else:
                    # Spans 2 bytes
                    if byte_pos + 1 < len(quant_data):
                        two_bytes = (quant_data[byte_pos] | (quant_data[byte_pos + 1] << 8))
                        q = (two_bytes >> bit_pos) & 0x3F
                    else:
                        q = 0
                
                bit_offset += 6
                
                # Convert to signed (-32 to 31)
                if q > 31:
                    q -= 64
                
                # Determine which scale to use (16 scales for 256 elements = 16 elements per scale)
                scale_idx = i // 16
                scale = float(scales[scale_idx])
                
                result[block_idx * QK_K + i] = q * scale
        
        return result.reshape(shape)
    
    def load_tensor(self, tensor_info, data_offset):
        """Load and dequantize a tensor"""
        name = tensor_info['name']
        shape = tensor_info['shape']
        dtype = tensor_info['dtype']
        offset = tensor_info['offset']
        
        # Seek to tensor data
        self.file.seek(data_offset + offset)
        
        # Calculate size based on dtype
        total_elements = int(np.prod(shape))
        
        if dtype == GGMLType.F32:
            # Read FP32 directly
            size = total_elements * 4
            data = np.frombuffer(self.file.read(size), dtype=np.float32)
            return data.reshape(shape)
        
        elif dtype == GGMLType.F16:
            # Read FP16 and convert to FP32
            size = total_elements * 2
            data = np.frombuffer(self.file.read(size), dtype=np.float16)
            return data.astype(np.float32).reshape(shape)
        
        elif dtype == GGMLType.Q4_0:
            # Q4_0 quantized
            block_size = 32
            num_blocks = (total_elements + block_size - 1) // block_size
            size = num_blocks * 18  # 2 bytes scale + 16 bytes data per block
            data = self.file.read(size)
            return self.dequantize_q4_0(data, shape)
        
        elif dtype == GGMLType.Q4_K:
            # Q4_K quantized (super-block format)
            QK_K = 256
            num_blocks = (total_elements + QK_K - 1) // QK_K
            size = num_blocks * 144  # 12 + 12 + 128 bytes per super-block
            data = self.file.read(size)
            return self.dequantize_q4_k(data, shape)
        
        elif dtype == GGMLType.Q6_K:
            # Q6_K quantized (super-block format)
            QK_K = 256
            num_blocks = (total_elements + QK_K - 1) // QK_K
            size = num_blocks * 224  # 32 + 192 bytes per super-block
            data = self.file.read(size)
            return self.dequantize_q6_k(data, shape)
        
        else:
            print(f"  ⚠️  Unsupported dtype {self.get_dtype_name(dtype)} for {name}")
            return None
    
    def extract_all(self):
        """Extract all tensors from GGUF file"""
        tensor_count, metadata_count = self.parse_header()
        self.parse_metadata(metadata_count)
        tensor_info_list = self.parse_tensor_info(tensor_count)
        
        # Calculate data offset (after header, metadata, and tensor info)
        data_offset = self.file.tell()
        
        # Align to 32 bytes
        alignment = 32
        data_offset = ((data_offset + alignment - 1) // alignment) * alignment
        
        print(f"\n[GGUF] Tensor data starts at offset: {data_offset}")
        print(f"[GGUF] Extracting tensors...")
        
        # Extract vision encoder tensors first
        vision_tensors = {}
        for i, info in enumerate(tensor_info_list):
            name = info['name']
            
            # Extract all tensors starting with 'v.' (vision encoder prefix in mllama)
            if name.startswith('v.'):
                print(f"  Loading [{i}/{len(tensor_info_list)}]: {name}")
                tensor = self.load_tensor(info, data_offset)
                if tensor is not None:
                    vision_tensors[name] = tensor
                    print(f"    ✅ Shape: {tensor.shape}, Dtype: {tensor.dtype}")
        
        print(f"\n✅ Extracted {len(vision_tensors)} vision tensors")
        return vision_tensors, self.metadata
    
    def close(self):
        self.file.close()

def main():
    print("="*80)
    print("GGUF to PyTorch Converter - Step 1: Parse GGUF")
    print("="*80)
    
    gguf_path = Path(r"C:\Users\Merlinthegrim\.ollama\models\blobs\sha256-9999d473417a8e179d993498195be5f42cab963acc75f4a6b15d981e8b68abed")
    
    if not gguf_path.exists():
        print(f"❌ GGUF file not found: {gguf_path}")
        return
    
    print(f"Input: {gguf_path}")
    print(f"Size: {gguf_path.stat().st_size / (1024**3):.2f} GB\n")
    
    reader = GGUFReader(gguf_path)
    
    try:
        tensors, metadata = reader.extract_all()
        
        # Save metadata
        output_dir = Path("D:/G.R.I.M/data/models/vision/llama3.2-vision-extracted")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "metadata.json", 'w') as f:
            # Convert to JSON-serializable format
            json_metadata = {k: str(v) if not isinstance(v, (int, float, str, bool, list)) else v 
                           for k, v in metadata.items()}
            json.dump(json_metadata, f, indent=2)
        
        print(f"\n✅ Metadata saved to: {output_dir / 'metadata.json'}")
        
        # Save tensors
        print(f"\n[Saving] Writing tensors to {output_dir}...")
        np.savez(output_dir / "vision_tensors.npz", **tensors)
        print(f"✅ Tensors saved to: {output_dir / 'vision_tensors.npz'}")
        
        print("\n" + "="*80)
        print("✅ Step 1 Complete!")
        print("="*80)
        print(f"Extracted {len(tensors)} vision tensors")
        print(f"Next: Run pytorch_reconstruction.py to build PyTorch model")
        
    finally:
        reader.close()

if __name__ == "__main__":
    main()

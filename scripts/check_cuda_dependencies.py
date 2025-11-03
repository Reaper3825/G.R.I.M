"""
Check CUDA dependencies for ONNX Runtime GPU support.
"""

import os
import sys
from pathlib import Path

def check_cuda():
    print("=" * 70)
    print("CUDA Dependency Check for ONNX Runtime GPU")
    print("=" * 70)
    print()
    
    # Check CUDA installation
    cuda_path = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if cuda_path.exists():
        cuda_versions = list(cuda_path.glob("v*"))
        if cuda_versions:
            print("✅ CUDA installed:")
            for v in cuda_versions:
                print(f"   - {v.name}")
            latest = sorted(cuda_versions)[-1]
            print(f"\n   Latest: {latest}")
            cuda_bin = latest / "bin"
        else:
            print("❌ CUDA directory exists but no versions found")
            return False
    else:
        print("❌ CUDA not installed")
        print(f"   Expected at: {cuda_path}")
        return False
    
    # Check for cuDNN
    print("\n" + "-" * 70)
    print("Checking for cuDNN...")
    print("-" * 70)
    
    cudnn_dlls = list(cuda_bin.glob("cudnn*.dll"))
    if cudnn_dlls:
        print("✅ cuDNN found:")
        for dll in cudnn_dlls:
            size_mb = dll.stat().st_size / (1024*1024)
            print(f"   - {dll.name} ({size_mb:.1f} MB)")
    else:
        print("❌ cuDNN NOT FOUND")
        print(f"   Searched in: {cuda_bin}")
        print()
        print("   To install cuDNN:")
        print("   1. Go to: https://developer.nvidia.com/cudnn")
        print("   2. Download cuDNN for CUDA 12.x")
        print("   3. Extract zip file")
        print(f"   4. Copy bin/*.dll to: {cuda_bin}")
        print(f"   5. Copy include/*.h to: {latest / 'include'}")
        print(f"   6. Copy lib/*.lib to: {latest / 'lib' / 'x64'}")
        print()
        print("   Required DLLs:")
        print("   - cudnn64_8.dll (or cudnn64_9.dll for newer versions)")
        print("   - cudnn_ops_infer64_8.dll")
        print("   - cudnn_cnn_infer64_8.dll")
        return False
    
    # Check vcpkg ONNX Runtime
    print("\n" + "-" * 70)
    print("Checking vcpkg ONNX Runtime GPU...")
    print("-" * 70)
    
    vcpkg_bin = Path(r"D:\G.R.I.M\vcpkg_installed\x64-windows\bin")
    onnx_dll = vcpkg_bin / "onnxruntime.dll"
    cuda_provider = vcpkg_bin / "onnxruntime_providers_cuda.dll"
    
    if onnx_dll.exists():
        print(f"✅ ONNX Runtime: {onnx_dll.name}")
        print(f"   Size: {onnx_dll.stat().st_size / (1024*1024):.1f} MB")
    else:
        print(f"❌ ONNX Runtime not found: {onnx_dll}")
        return False
    
    if cuda_provider.exists():
        print(f"✅ CUDA Provider: {cuda_provider.name}")
        print(f"   Size: {cuda_provider.stat().st_size / (1024*1024):.1f} MB")
    else:
        print(f"❌ CUDA Provider not found: {cuda_provider}")
        return False
    
    # Check PATH
    print("\n" + "-" * 70)
    print("Checking PATH for CUDA...")
    print("-" * 70)
    
    path_dirs = os.environ.get('PATH', '').split(';')
    cuda_in_path = [p for p in path_dirs if 'cuda' in p.lower()]
    
    if cuda_in_path:
        print("✅ CUDA in PATH:")
        for p in cuda_in_path:
            print(f"   - {p}")
    else:
        print("⚠️  CUDA not in PATH")
        print(f"   Add to PATH: {cuda_bin}")
    
    print("\n" + "=" * 70)
    if cudnn_dlls:
        print("✅ All dependencies met! CUDA should work in C++.")
        print("=" * 70)
        print("\nExpected performance:")
        print("  CPU: ~8200ms")
        print("  GPU: ~100-300ms (27x faster)")
        return True
    else:
        print("⚠️  cuDNN missing - using CPU fallback")
        print("=" * 70)
        print("\nCurrent performance:")
        print("  CPU: ~8200ms (working)")
        print()
        print("To enable GPU acceleration, install cuDNN (see instructions above)")
        return False

if __name__ == "__main__":
    success = check_cuda()
    sys.exit(0 if success else 1)

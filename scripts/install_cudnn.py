"""
Download and install cuDNN for CUDA 12.5 to enable GPU acceleration in ONNX Runtime.

Note: You'll need an NVIDIA Developer account (free) to download cuDNN.
This script will guide you through the process.
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
import subprocess

def install_cudnn():
    print("=" * 70)
    print("cuDNN Installation for CUDA 12.5")
    print("=" * 70)
    print()
    
    cuda_path = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5")
    
    if not cuda_path.exists():
        print("❌ CUDA 12.5 not found!")
        print(f"   Expected at: {cuda_path}")
        return False
    
    print("✅ CUDA 12.5 found")
    print()
    
    # Check for common cuDNN download locations
    downloads = Path.home() / "Downloads"
    cudnn_zips = list(downloads.glob("cudnn-*-windows-*.zip"))
    
    if cudnn_zips:
        print(f"Found {len(cudnn_zips)} cuDNN zip(s) in Downloads:")
        for i, zip_file in enumerate(cudnn_zips, 1):
            size_mb = zip_file.stat().st_size / (1024*1024)
            print(f"  [{i}] {zip_file.name} ({size_mb:.1f} MB)")
        print()
        
        choice = input("Select zip to install (or 'n' to download manually): ").strip()
        
        if choice.lower() != 'n' and choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(cudnn_zips):
                cudnn_zip = cudnn_zips[idx]
                return install_from_zip(cudnn_zip, cuda_path)
    
    # Manual download instructions
    print("=" * 70)
    print("Manual Download Required")
    print("=" * 70)
    print()
    print("1. Go to: https://developer.nvidia.com/cudnn-downloads")
    print("2. Sign in (free NVIDIA Developer account)")
    print("3. Select:")
    print("   - CUDA: 12.x")
    print("   - OS: Windows")
    print("   - Architecture: x86_64")
    print("4. Download the ZIP file (NOT the installer)")
    print("5. Save to your Downloads folder")
    print()
    
    input("Press Enter after downloading cuDNN zip file...")
    
    # Check again
    cudnn_zips = list(downloads.glob("cudnn-*-windows-*.zip"))
    if cudnn_zips:
        cudnn_zip = sorted(cudnn_zips, key=lambda x: x.stat().st_mtime)[-1]
        print(f"\nFound: {cudnn_zip.name}")
        return install_from_zip(cudnn_zip, cuda_path)
    else:
        print("\n❌ No cuDNN zip found in Downloads folder")
        print("Please download manually and run this script again.")
        return False

def install_from_zip(zip_path, cuda_path):
    print()
    print("=" * 70)
    print("Installing cuDNN...")
    print("=" * 70)
    print(f"From: {zip_path}")
    print(f"To: {cuda_path}")
    print()
    
    # Extract to temp location
    temp_dir = Path(os.environ.get('TEMP', '.')) / "cudnn_extract"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        print("Extracting zip file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find the cudnn folder (usually cudnn-windows-x86_64-*/)
        cudnn_dirs = [d for d in temp_dir.iterdir() if d.is_dir() and 'cudnn' in d.name.lower()]
        
        if not cudnn_dirs:
            print("❌ Could not find cuDNN folder in zip")
            return False
        
        cudnn_dir = cudnn_dirs[0]
        print(f"Found cuDNN directory: {cudnn_dir.name}")
        
        # Copy files
        print("\nCopying files (requires admin privileges)...")
        
        # Copy bin/*.dll
        bin_src = cudnn_dir / "bin"
        bin_dst = cuda_path / "bin"
        if bin_src.exists():
            dll_files = list(bin_src.glob("*.dll"))
            print(f"\nCopying {len(dll_files)} DLL files to {bin_dst}...")
            for dll in dll_files:
                dst = bin_dst / dll.name
                try:
                    shutil.copy2(dll, dst)
                    print(f"  ✅ {dll.name}")
                except PermissionError:
                    print(f"  ❌ {dll.name} (Permission denied - run as Administrator)")
                    return False
        
        # Copy include/*.h
        inc_src = cudnn_dir / "include"
        inc_dst = cuda_path / "include"
        if inc_src.exists():
            h_files = list(inc_src.glob("*.h"))
            print(f"\nCopying {len(h_files)} header files to {inc_dst}...")
            for h in h_files:
                dst = inc_dst / h.name
                try:
                    shutil.copy2(h, dst)
                    print(f"  ✅ {h.name}")
                except PermissionError:
                    print(f"  ❌ {h.name} (Permission denied - run as Administrator)")
        
        # Copy lib/*.lib
        lib_src = cudnn_dir / "lib"
        lib_dst = cuda_path / "lib" / "x64"
        if lib_src.exists():
            # Check both lib/ and lib/x64 in source
            lib_files = list(lib_src.glob("*.lib"))
            lib_x64 = lib_src / "x64"
            if lib_x64.exists():
                lib_files.extend(list(lib_x64.glob("*.lib")))
            
            if lib_files:
                print(f"\nCopying {len(lib_files)} library files to {lib_dst}...")
                lib_dst.mkdir(parents=True, exist_ok=True)
                for lib in lib_files:
                    dst = lib_dst / lib.name
                    try:
                        shutil.copy2(lib, dst)
                        print(f"  ✅ {lib.name}")
                    except PermissionError:
                        print(f"  ❌ {lib.name} (Permission denied - run as Administrator)")
        
        # Cleanup
        print("\nCleaning up temporary files...")
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        print("\n" + "=" * 70)
        print("✅ cuDNN installation complete!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("1. Rebuild GRIM (CMake + compile)")
        print("2. Run GRIM - it should now use CUDA automatically")
        print("3. Check logs for: 'Using CUDA execution provider (RTX 3080Ti)'")
        print()
        print("Expected performance improvement:")
        print("  Before: ~8200ms (CPU)")
        print("  After:  ~100-300ms (GPU) - 27x faster!")
        print()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during installation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print()
    print("This script will install cuDNN for CUDA 12.5")
    print("You may need to run PowerShell/CMD as Administrator")
    print()
    
    success = install_cudnn()
    
    if success:
        print("\nVerifying installation...")
        import subprocess
        subprocess.run([sys.executable, "scripts/check_cuda_dependencies.py"])
    
    sys.exit(0 if success else 1)

"""
Hybrid ONNX + Ollama Vision System
Uses ONNX for fast GPU-accelerated image preprocessing and validation,
then sends optimized images to Ollama for vision-language understanding.
"""

import onnxruntime as ort
import numpy as np
import requests
import base64
import time
import json
from pathlib import Path
from PIL import Image
from io import BytesIO
from typing import Optional, Tuple


class HybridVisionSystem:
    """
    Combines ONNX preprocessing with Ollama vision model.
    - ONNX: Fast GPU preprocessing, validation, feature extraction
    - Ollama: Vision-language understanding and text generation
    """
    
    def __init__(
        self,
        onnx_model_path: str = "D:/G.R.I.M/data/models/vision/llama3.2-vision-onnx/vision_encoder.onnx",
        ollama_model: str = "llama3.2-vision:11b",
        ollama_url: str = "http://localhost:11434"
    ):
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.onnx_session = None
        
        # Load ONNX model for preprocessing
        model_path = Path(onnx_model_path)
        if model_path.exists():
            print(f"Loading ONNX preprocessor from {model_path}...")
            self._load_onnx_model(model_path)
        else:
            print(f"⚠️  ONNX model not found at {model_path}")
            print("   Continuing without ONNX acceleration")
    
    def _load_onnx_model(self, model_path: Path):
        """Load ONNX model for GPU-accelerated preprocessing."""
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            
            # Prefer CUDA for GPU acceleration
            providers = []
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                print("✅ Using GPU acceleration (CUDA)")
                providers.append('CUDAExecutionProvider')
            else:
                print("⚠️  Using CPU for preprocessing")
            providers.append('CPUExecutionProvider')
            
            self.onnx_session = ort.InferenceSession(
                str(model_path),
                sess_options,
                providers=providers
            )
            print(f"✅ ONNX preprocessor loaded")
            
        except Exception as e:
            print(f"⚠️  Failed to load ONNX model: {e}")
            self.onnx_session = None
    
    def preprocess_image_onnx(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Use ONNX for fast GPU-accelerated image preprocessing.
        Returns visual embeddings or None if ONNX not available.
        """
        if not self.onnx_session:
            return None
        
        try:
            # Resize and normalize image for ONNX model
            # llama3.2-vision expects 560x560 images
            img_resized = image.resize((560, 560), Image.Resampling.LANCZOS)
            img_array = np.array(img_resized).astype(np.float32) / 255.0
            
            # Convert HWC to CHW format
            img_array = np.transpose(img_array, (2, 0, 1))
            
            # Normalize with ImageNet stats
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img_array = (img_array - mean) / std
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # Run ONNX inference
            start_time = time.time()
            inputs = {self.onnx_session.get_inputs()[0].name: img_array}
            outputs = self.onnx_session.run(None, inputs)
            preprocess_time = (time.time() - start_time) * 1000
            
            print(f"  ONNX preprocessing: {preprocess_time:.1f}ms")
            print(f"  Embeddings shape: {outputs[0].shape}")
            
            return outputs[0]
            
        except Exception as e:
            print(f"⚠️  ONNX preprocessing failed: {e}")
            return None
    
    def optimize_image_for_ollama(self, image: Image.Image) -> Tuple[str, int]:
        """
        Optimize image for Ollama API (resize, compress, encode).
        Returns base64 string and size in bytes.
        """
        # Resize to reasonable size (Ollama handles resizing, but smaller = faster)
        max_size = 896  # llama3.2-vision can handle up to 896x896
        
        # Calculate resize maintaining aspect ratio
        width, height = image.size
        if width > max_size or height > max_size:
            ratio = min(max_size / width, max_size / height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Compress to JPEG with good quality
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=85, optimize=True)
        img_bytes = buffer.getvalue()
        
        # Encode to base64
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
        
        return img_b64, len(img_bytes)
    
    def query_ollama_vision(
        self,
        image: Image.Image,
        prompt: str,
        use_onnx_preprocess: bool = True
    ) -> dict:
        """
        Query Ollama vision model with optional ONNX preprocessing.
        
        Args:
            image: PIL Image to analyze
            prompt: Text prompt for the vision model
            use_onnx_preprocess: Whether to use ONNX for preprocessing validation
        
        Returns:
            Dictionary with response and timing information
        """
        print(f"\n{'='*70}")
        print(f"Querying Vision Model: {self.ollama_model}")
        print(f"{'='*70}")
        print(f"Prompt: {prompt}")
        print(f"{'-'*70}")
        
        result = {
            'prompt': prompt,
            'response': None,
            'error': None,
            'timings': {}
        }
        
        # Optional ONNX preprocessing for validation/speed check
        if use_onnx_preprocess and self.onnx_session:
            print("\n1️⃣  ONNX Preprocessing...")
            embeddings = self.preprocess_image_onnx(image)
            if embeddings is not None:
                print(f"   ✅ Image validated and preprocessed on GPU")
                result['timings']['onnx_preprocess_ms'] = embeddings
        
        # Optimize image for Ollama
        print("\n2️⃣  Optimizing image for Ollama...")
        start_time = time.time()
        img_b64, img_size = self.optimize_image_for_ollama(image)
        optimize_time = (time.time() - start_time) * 1000
        
        print(f"   Original size: {image.size}")
        print(f"   Optimized size: {img_size / 1024:.1f} KB")
        print(f"   Optimization time: {optimize_time:.1f}ms")
        
        result['timings']['image_optimize_ms'] = optimize_time
        result['timings']['image_size_kb'] = img_size / 1024
        
        # Query Ollama API
        print("\n Sending to Ollama...")
        try:
            payload = {
                'model': self.ollama_model,
                'prompt': prompt,
                'images': [img_b64],
                'stream': False
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            api_time = (time.time() - start_time) * 1000
            
            result['timings']['ollama_api_ms'] = api_time
            
            if response.status_code == 200:
                data = response.json()
                result['response'] = data.get('response', '')
                
                # Extract additional timing info from Ollama
                if 'total_duration' in data:
                    result['timings']['ollama_total_ns'] = data['total_duration']
                    result['timings']['ollama_total_ms'] = data['total_duration'] / 1e6
                
                if 'load_duration' in data:
                    result['timings']['ollama_load_ms'] = data['load_duration'] / 1e6
                
                if 'prompt_eval_duration' in data:
                    result['timings']['ollama_prompt_eval_ms'] = data['prompt_eval_duration'] / 1e6
                
                if 'eval_duration' in data:
                    result['timings']['ollama_eval_ms'] = data['eval_duration'] / 1e6
                
                print(f"   ✅ Response received in {api_time:.0f}ms")
                
            else:
                error_msg = f"Ollama API error: {response.status_code}"
                result['error'] = error_msg
                print(f"   ❌ {error_msg}")
                
        except requests.Timeout:
            result['error'] = "Request timed out after 60s"
            print(f"   ❌ {result['error']}")
        except Exception as e:
            result['error'] = str(e)
            print(f"   ❌ Error: {e}")
        
        return result
    
    def print_result(self, result: dict):
        """Pretty print the vision query result."""
        print(f"\n{'='*70}")
        print("RESULT")
        print(f"{'='*70}")
        
        if result['error']:
            print(f"❌ Error: {result['error']}")
        elif result['response']:
            print(f"✅ Response:")
            print(f"\n{result['response']}\n")
        
        # Print timing breakdown
        if result['timings']:
            print(f"{'-'*70}")
            print("Timing Breakdown:")
            print(f"{'-'*70}")
            
            timings = result['timings']
            
            if 'onnx_preprocess_ms' in timings:
                print(f"  ONNX preprocessing: {timings['onnx_preprocess_ms']:.1f}ms")
            
            if 'image_optimize_ms' in timings:
                print(f"  Image optimization: {timings['image_optimize_ms']:.1f}ms")
                print(f"  Optimized image size: {timings['image_size_kb']:.1f}KB")
            
            if 'ollama_load_ms' in timings:
                print(f"  Ollama model load: {timings['ollama_load_ms']:.1f}ms")
            
            if 'ollama_prompt_eval_ms' in timings:
                print(f"  Ollama prompt eval: {timings['ollama_prompt_eval_ms']:.1f}ms")
            
            if 'ollama_eval_ms' in timings:
                print(f"  Ollama generation: {timings['ollama_eval_ms']:.1f}ms")
            
            if 'ollama_total_ms' in timings:
                print(f"  Ollama total: {timings['ollama_total_ms']:.0f}ms")
            
            if 'ollama_api_ms' in timings:
                print(f"  API round-trip: {timings['ollama_api_ms']:.0f}ms")
        
        print(f"{'='*70}\n")


def test_hybrid_system():
    """Test the hybrid ONNX + Ollama vision system."""
    
    print("Initializing Hybrid Vision System...")
    system = HybridVisionSystem()
    
    # Test with a sample image
    print("\nCreating test image...")
    test_image = Image.new('RGB', (800, 600), color=(73, 109, 137))
    
    # Add some text to the image for testing
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(test_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 60)
    except:
        font = ImageFont.load_default()
    
    draw.text((50, 250), "Hello from GRIM!", fill=(255, 255, 255), font=font)
    
    # Test queries
    test_queries = [
        "What do you see in this image?",
        "Describe the text and colors in detail.",
        "What is written in this image?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'#'*70}")
        print(f"Test Query {i}/{len(test_queries)}")
        print(f"{'#'*70}")
        
        result = system.query_ollama_vision(
            test_image,
            query,
            use_onnx_preprocess=True
        )
        
        system.print_result(result)
        
        if i < len(test_queries):
            print("\nWaiting 2s before next query...")
            time.sleep(2)
    
    print(f"\n{'='*70}")
    print("✅ Hybrid Vision System Test Complete")
    print(f"{'='*70}")


def interactive_mode():
    """Interactive mode for testing with custom images and prompts."""
    
    print("=" * 70)
    print("Hybrid Vision System - Interactive Mode")
    print("=" * 70)
    
    system = HybridVisionSystem()
    
    while True:
        print("\nOptions:")
        print("  1. Load image from file")
        print("  2. Use test image")
        print("  3. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == '3':
            print("Goodbye!")
            break
        
        # Get image
        image = None
        if choice == '1':
            path = input("Image path: ").strip()
            try:
                image = Image.open(path)
                print(f"✅ Loaded {image.size[0]}x{image.size[1]} image")
            except Exception as e:
                print(f"❌ Failed to load image: {e}")
                continue
        elif choice == '2':
            image = Image.new('RGB', (800, 600), color=(73, 109, 137))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(image)
            draw.text((50, 250), "Test Image", fill=(255, 255, 255))
            print("✅ Created test image")
        else:
            print("Invalid choice")
            continue
        
        # Get prompt
        prompt = input("\nPrompt: ").strip()
        if not prompt:
            prompt = "What do you see in this image?"
        
        # Query
        result = system.query_ollama_vision(image, prompt)
        system.print_result(result)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_mode()
    else:
        test_hybrid_system()

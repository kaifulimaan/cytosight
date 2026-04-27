#!/usr/bin/env python3
"""
Quick test script to trigger a prediction and see full logs.
Creates a dummy tissue image and runs through diagnosis endpoint.
"""

import sys
import os
import requests
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

import numpy as np
from PIL import Image
import io

# Configuration
API_URL = "http://localhost:8000/api"
TEST_IMAGE_SIZE = (224, 224)

def create_dummy_tissue_image() -> bytes:
    """Create a simple synthetic tissue-like image for testing."""
    # Create random tissue-like image with variation
    img_array = np.random.randint(50, 200, (TEST_IMAGE_SIZE[0], TEST_IMAGE_SIZE[1], 3), dtype=np.uint8)
    
    # Add some structure to make it more realistic
    for _ in range(10):
        y, x = np.random.randint(0, TEST_IMAGE_SIZE[0], 2)
        radius = np.random.randint(5, 30)
        yy, xx = np.ogrid[:TEST_IMAGE_SIZE[0], :TEST_IMAGE_SIZE[1]]
        mask = (yy - y)**2 + (xx - x)**2 <= radius**2
        img_array[mask] = np.random.randint(30, 180, 3, dtype=np.uint8)
    
    # Convert to PIL Image
    img = Image.fromarray(img_array, 'RGB')
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.getvalue()

def test_diagnosis():
    """Test diagnosis prediction endpoint."""
    print("\n" + "="*80)
    print("🧪 TESTING DIAGNOSIS PREDICTION")
    print("="*80 + "\n")
    
    # Create dummy image
    print("1️⃣  Creating dummy tissue image...")
    image_bytes = create_dummy_tissue_image()
    print(f"   ✅ Image created: {len(image_bytes)} bytes\n")
    
    # Prepare request
    print("2️⃣  Preparing diagnosis request...")
    
    files = {'file': ('test.jpg', image_bytes, 'image/jpeg')}
    
    # Call diagnosis endpoint (using upload-and-predict which requires no auth)
    print("3️⃣  Calling /api/diagnosis/upload-and-predict endpoint...")
    print(f"   URL: {API_URL}/diagnosis/upload-and-predict\n")
    
    try:
        response = requests.post(
            f"{API_URL}/diagnosis/upload-and-predict",
            files=files,
            timeout=60
        )
        
        print(f"\n   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ PREDICTION SUCCESSFUL!\n")
            print(json.dumps(result, indent=2))
        else:
            print(f"\n❌ ERROR: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("📊 CHECK BACKEND TERMINAL FOR DETAILED LOGS:")
    print("   - Disease order mapping")
    print("   - Raw prediction logits")
    print("   - Label mappings")
    print("="*80 + "\n")

if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                      CYTOSIGHT PREDICTION TEST                            ║
    ║                                                                            ║
    ║  This script will:                                                        ║
    ║  1. Create a dummy tissue image                                           ║
    ║  2. Send it to the diagnosis endpoint                                     ║
    ║  3. Print the response                                                    ║
    ║                                                                            ║
    ║  🔄 WATCH THE BACKEND TERMINAL FOR FULL LOGS!                             ║
    ║     Look for the emoji-decorated log lines showing:                       ║
    ║     - Disease order [0][1][2]...                                          ║
    ║     - Raw logits and probabilities                                        ║
    ║     - Label mapping results                                               ║
    ║                                                                            ║
    ║  Terminal ID: 48f1e260-090b-4bf9-9017-686ad61d2015                        ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    test_diagnosis()

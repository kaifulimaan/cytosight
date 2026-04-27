#!/usr/bin/env python3
"""Test upload endpoint to diagnose 404 not_found error."""

import sys
import os
import requests
import json
from pathlib import Path
import io
from PIL import Image
import numpy as np

# Configuration
API_URL = "http://localhost:8000/api"

def get_test_token():
    """Get a valid test token from Supabase (requires SUPABASE credentials)."""
    # For now, return a dummy - the error will show us if auth is the issue
    return "test_token_placeholder"

def create_dummy_image() -> bytes:
    """Create a simple test image."""
    img_array = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.getvalue()

def test_upload():
    """Test the image upload endpoint."""
    print("\n" + "="*80)
    print("📤 TESTING IMAGE UPLOAD")
    print("="*80 + "\n")
    
    # Create test image
    print("1️⃣  Creating test image...")
    image_bytes = create_dummy_image()
    print(f"   ✅ Image created: {len(image_bytes)} bytes\n")
    
    # Prepare upload
    files = {'file': ('test.jpg', image_bytes, 'image/jpeg')}
    headers = {
        'Authorization': 'Bearer test_token_placeholder'
    }
    
    print("2️⃣  Uploading to /api/upload/image...")
    print(f"   URL: {API_URL}/upload/image")
    print(f"   Headers: {list(headers.keys())}\n")
    
    try:
        response = requests.post(
            f"{API_URL}/upload/image",
            files=files,
            headers=headers,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}\n")
        
        if response.status_code == 200:
            print("✅ UPLOAD SUCCESSFUL!\n")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ UPLOAD FAILED\n")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"❌ REQUEST ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Check backend terminal for detailed DEBUG logs")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_upload()

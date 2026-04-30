import time
import random
import string
from playwright.sync_api import sync_playwright
from PIL import Image
import io
import os

# Configuration
BASE_URL = "http://localhost:8080"  # Your local frontend port is 8080
TEST_PASSWORD = "Password123!"

def generate_random_email():
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"test_{random_str}@example.com"

def create_test_image(path="test_sample.jpg"):
    """Creates a simple dummy image for testing upload."""
    img = Image.new('RGB', (224, 224), color = (73, 109, 137))
    img.save(path)
    return os.path.abspath(path)

def run_test():
    email = generate_random_email()
    test_image = create_test_image()
    
    with sync_playwright() as p:
        print("\n" + "="*50)
        print("🚀 STARTING CYTOSIGHT E2E TEST SUITE")
        print("="*50 + "\n")
        
        # Launch browser (headless=False so you can watch it!)
        # slow_mo adds a small delay between actions so it's visible
        browser = p.chromium.launch(headless=False, slow_mo=800) 
        context = browser.new_context()
        page = context.new_page()

        try:
            # --- 1. SIGNUP ---
            print(f"📝 STEP 1: Signing up as {email}...")
            page.goto(f"{BASE_URL}/signup")
            page.fill('input[name="fullName"]', "Test User")
            page.fill('input[name="email"]', email)
            page.fill('input[name="password"]', TEST_PASSWORD)
            page.fill('input[name="confirmPassword"]', TEST_PASSWORD)
            page.click('button[type="submit"]')
            
            # Wait for dashboard redirect
            page.wait_for_url("**/dashboard", timeout=10000)
            print("   ✅ Signup Successful! Redirected to Dashboard.\n")

            # --- 2. UPLOAD & DIAGNOSIS ---
            print("📤 STEP 2: Testing Image Upload...")
            page.goto(f"{BASE_URL}/upload")
            
            # Upload the file (handles the hidden input automatically)
            page.set_input_files('input[type="file"]', test_image)
            print(f"   ✅ Image selected: {test_image}")
            
            # Click Diagnosis
            print("🔍 STEP 3: Running Diagnosis (hitting HF Backend)...")
            page.click('button:has-text("Disease Diagnosis")')
            
            # Wait for Results Page (might take a few seconds for AI to process)
            print("   ⏳ Waiting for AI results (up to 60s)...")
            page.wait_for_selector('text=Diagnosis Completed', timeout=60000)
            print("   ✅ Diagnosis Successful!\n")

            # --- 3. UPLOAD & SEGMENTATION ---
            print("📤 STEP 4: Testing Binary Segmentation...")
            page.goto(f"{BASE_URL}/upload")
            
            # Upload the file again
            page.set_input_files('input[type="file"]', test_image)
            print(f"   ✅ Image selected for segmentation")
            
            # Click Segmentation
            print("🎭 STEP 5: Running Segmentation (hitting HF Backend)...")
            page.click('button:has-text("Binary Segmentation")')
            
            # Wait for Results Page
            print("   ⏳ Waiting for Segmentation results (up to 60s)...")
            page.wait_for_selector('text=Segmentation Results', timeout=60000)
            print("   ✅ Segmentation Successful!")

            print("\n" + "*"*50)
            print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
            print("*"*50 + "\n")
            
        except Exception as e:
            print(f"\n❌ TEST FAILED: {str(e)}")
            # Take a screenshot on failure to see what happened
            page.screenshot(path="test_failure.png")
            print("📸 Screenshot of failure saved to 'test_failure.png'")
            raise e
            
        finally:
            # Clean up
            print("🧹 Cleaning up...")
            time.sleep(3) # Give you a moment to see the success
            browser.close()
            if os.path.exists(test_image):
                os.remove(test_image)

if __name__ == "__main__":
    run_test()

import time
import json
import os
from playwright.sync_api import sync_playwright

# Configuration
BASE_URL = "http://localhost:8080"
# Use specific credentials provided by the user
TEST_EMAIL = "kaifulimaan@gmail.com"
TEST_PASSWORD = "Kk123456,"
REAL_IMAGE_PATH = r"D:\Projects\CytoSight\cytosight\test_sampe.png"
EVIDENCE_DIR = "test_evidence"

if not os.path.exists(EVIDENCE_DIR):
    os.makedirs(EVIDENCE_DIR)


def run_test():
    if not os.path.exists(REAL_IMAGE_PATH):
        print(f"❌ ERROR: Real image not found at {REAL_IMAGE_PATH}")
        return

    with sync_playwright() as p:
        print("\n" + "="*50)
        print("🚀 STARTING CYTOSIGHT EVIDENCE-BASED PERFORMANCE TEST")
        print("="*50 + "\n")
        
        browser = p.chromium.launch(headless=False, slow_mo=500)
        context = browser.new_context()
        page = context.new_page()

        # Storage for performance metrics
        request_start_times = {}

        def handle_request(request):
            if "/api/" in request.url:
                request_start_times[request.url] = time.time()

        def handle_response(response):
            url = response.url
            if "/api/" in url and response.status in [200, 201]:
                try:
                    # Calculate duration
                    start_time = request_start_times.get(url)
                    duration = round(time.time() - start_time, 2) if start_time else None
                    
                    # Capture JSON data
                    data = response.json()
                    
                    # Add Performance Metadata
                    data["_performance_metrics"] = {
                        "total_response_time_seconds": duration,
                        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    }

                    # Determine specific category for naming
                    category = "API_RESPONSE"
                    if "auth/login" in url:
                        category = "LOGIN"
                    elif "diagnosis/history" in url:
                        category = "HISTORY_LOAD"
                    elif "upload" in url:
                        category = "IMAGE_UPLOAD"
                    elif "diagnosis/predict" in url:
                        category = "DIAGNOSIS_MODEL"
                    elif "segmentation/predict" in url:
                        category = "SEGMENTATION_MODEL"
                    elif "explain" in url:
                        category = "EXPLAINABLE_AI"

                    filename = f"{EVIDENCE_DIR}/{category}_{int(time.time())}.json"
                    with open(filename, "w") as f:
                        json.dump(data, f, indent=4)
                    
                    print(f"   📥 Captured {category} Response ({duration}s): {filename}")
                except Exception:
                    pass

        page.on("request", handle_request)
        page.on("response", handle_response)

        try:
            # --- 1. LOGIN ---
            print(f"🔐 STEP 1: Logging in as {TEST_EMAIL}...")
            page.goto(f"{BASE_URL}/login")
            
            page.fill('input[name="email"]', TEST_EMAIL)
            page.fill('input[name="password"]', TEST_PASSWORD)
            
            page.click('button[type="submit"]')
            
            # Wait for dashboard to load
            page.wait_for_selector('text=Welcome back', timeout=30000)
            print("   ✅ Login Successful!\n")

            # --- 2. HISTORY ---
            print("📜 STEP 2: Loading History Page...")
            page.click('text=View Past Diagnosis') # Navigate from dashboard
            page.wait_for_selector('text=Diagnosis History', timeout=30000)
            # Give it a second to fetch history
            time.sleep(2) 
            print("   ✅ History Loaded.\n")

            # --- 3. UPLOAD & DIAGNOSIS ---
            print("📤 STEP 3: Running Diagnosis...")
            page.goto(f"{BASE_URL}/upload")
            
            page.set_input_files('input[type="file"]', REAL_IMAGE_PATH)
            print(f"   ✅ Image Selected: {REAL_IMAGE_PATH}")
            
            page.click('button:has-text("Disease Diagnosis")')
            print("   ⏳ Processing Diagnosis...")
            
            page.wait_for_selector('text=Diagnosis Completed', timeout=120000)
            print("   ✅ Diagnosis Completed.\n")

            # --- 4. UPLOAD & SEGMENTATION ---
            print("📤 STEP 4: Running Segmentation...")
            page.goto(f"{BASE_URL}/upload")
            
            page.set_input_files('input[type="file"]', REAL_IMAGE_PATH)
            page.click('button:has-text("Binary Segmentation")')
            print("   ⏳ Processing Segmentation...")
            
            page.wait_for_selector('text=Segmentation Results', timeout=120000)
            print("   ✅ Segmentation Completed.\n")

            print("*"*50)
            print("🎉 ALL TESTS PASSED. PERFORMANCE DATA SAVED.")
            print(f"📁 Evidence available in: {os.path.abspath(EVIDENCE_DIR)}")
            print("*"*50 + "\n")
            
        except Exception as e:
            print(f"\n❌ TEST FAILED: {str(e)}")
            # Optional: capture error screenshot if it failed
            # page.screenshot(path=f"{EVIDENCE_DIR}/ERROR_STATE.png")
            raise e
            
        finally:
            print("🧹 Cleaning up...")
            time.sleep(2)
            browser.close()

if __name__ == "__main__":
    run_test()

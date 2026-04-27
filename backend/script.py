import os
import requests
from tqdm import tqdm
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor

URL = r"https://yvkmaiiycqligmloxwbx.supabase.co"
KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inl2a21haWl5Y3FsaWdtbG94d2J4Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MDQ1MDQ2NCwiZXhwIjoyMDg2MDI2NDY0fQ.gxBw1bzwDuA1tTwDAoekHArpvhzNHIuOdZxJaPy7lnI"
BUCKET = "diagnostic-model"
LOCAL_FILE = r"D:\Projects\CytoSight\cytosight\backend\app\phikonv2\model.safetensors"
REMOTE_PATH = "v1/model.safetensors"

def upload_with_fast_failure():
    file_size = os.path.getsize(LOCAL_FILE)
    upload_url = f"{URL}/storage/v1/object/{BUCKET}/{REMOTE_PATH}"
    
    print(f"🚀 Preparing Upload: {os.path.basename(LOCAL_FILE)} ({file_size / (1024**3):.2f} GB)")

    with open(LOCAL_FILE, 'rb') as f:
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="📡 Uploading")

        def callback(monitor):
            pbar.update(monitor.bytes_read - pbar.n)

        encoder = MultipartEncoder(fields={'file': (os.path.basename(LOCAL_FILE), f, 'application/octet-stream')})
        monitor = MultipartEncoderMonitor(encoder, callback)

        headers = {
            'Authorization': f'Bearer {KEY}',
            'Content-Type': monitor.content_type,
            'x-upsert': 'true',
            'Expect': '100-continue'  
        }

        try:

            session = requests.Session()
            response = session.post(upload_url, data=monitor, headers=headers, timeout=None)
            pbar.close()

            if response.status_code in [200, 201]:
                print("\n✅ Upload successful!")
            else:

                print(f"\n❌ Server Rejected Upload: {response.status_code}")
                print(f"Message: {response.text}")
                
        except requests.exceptions.RequestException as e:
            pbar.close()
            print(f"\n❌ Network Error: {e}")

if __name__ == "__main__":
    upload_with_fast_failure()
// ===== ENVIRONMENT-AWARE API URL DETECTION =====
function getBaseUrl(): string {
  // Priority 1: Check localStorage for user-set ngrok URL
  const storedBackendUrl = localStorage.getItem('backend_url');
  if (storedBackendUrl) {
    console.log('✅ Using stored backend URL:', storedBackendUrl);
    return storedBackendUrl;
  }

  // Priority 2: Check environment variable
  if (typeof import.meta.env.VITE_BACKEND_URL !== 'undefined' && import.meta.env.VITE_BACKEND_URL) {
    const envUrl = import.meta.env.VITE_BACKEND_URL;
    // Add /api suffix if not present
    const finalUrl = envUrl.endsWith('/api') ? envUrl : `${envUrl.replace(/\/$/, '')}/api`;
    console.log('✅ Using VITE_BACKEND_URL:', finalUrl);
    return finalUrl;
  }

  // Priority 3: Check if running on local development
  const hostname = window.location.hostname;
  const isLocalhost = hostname === 'localhost' || hostname === '127.0.0.1';
  const isLocalDomain = hostname.includes('local') || hostname.includes('10.7.');

  if (isLocalhost || isLocalDomain) {
    console.log('✅ Running on local, using localhost:8000/api');
    return 'http://localhost:8000/api';
  }

  // Priority 4: Production fallback
  console.log('☁️ Running on production domain - using Hugging Face backend');
  return 'https://kimaan28-cytosight.hf.space/api';
}

function getCurrentBaseUrl(): string {
  return getBaseUrl();
}

export function setBackendUrl(url: string) {
  localStorage.setItem('backend_url', url);
  console.log('✅ Backend URL updated to:', url);
  console.log('🔄 Refresh page or make API call for changes to take effect');
}

type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";

export async function apiCall<T = unknown>(
  method: HttpMethod,
  endpoint: string,
  body?: unknown,
  token?: string
): Promise<T> {
  const authToken = token ?? getToken();

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };

  if (authToken) {
    headers.Authorization = `Bearer ${authToken}`;
  }

  const baseUrl = getCurrentBaseUrl();
  const url = `${baseUrl}${endpoint}`;
  
  try {
    console.log(`[API Request] ${method} ${url}`);
    
    const res = await fetch(url, {
      method,
      headers,
      credentials: 'include',
      body: body === undefined ? undefined : JSON.stringify(body),
    });

    console.log(`[API Response] ${method} ${url} - Status ${res.status}`);

    if (!res.ok) {
      let errorMessage = `HTTP ${res.status}`;
      
      try {
        const error = await res.json();
        errorMessage = error.detail || errorMessage;
      } catch {
        errorMessage = res.statusText || errorMessage;
      }

      if (res.status === 401) {
        errorMessage = "Unauthorized - Please login again";
      } else if (res.status === 403) {
        errorMessage = "Forbidden - You don't have permission";
      } else if (res.status === 404) {
        errorMessage = "Resource not found";
      } else if (res.status === 500) {
        errorMessage = "Server error - Backend may not be running";
      }

      console.error(`[API Error] ${errorMessage}`);
      throw new Error(errorMessage);
    }

    if (res.status === 204) {
      return undefined as T;
    }

    const data = await res.json();
    console.log(`[API Success] ${method} ${url}`);
    return data;
  } catch (error) {
    if (error instanceof TypeError) {
      if (error.message.includes('Failed to fetch')) {
        const helpMsg = 'Failed to reach backend. Make sure:\n1. Backend is running\n2. ngrok tunnel is active\n3. ngrok URL is correct in localStorage';
        console.error(`[CORS/Network Error] ${helpMsg}`);
        throw new Error(helpMsg);
      }
    }
    
    console.error(`[API Error] ${method} ${url}`, error);
    throw error;
  }
}

// ✅ FIXED: signup and login now use proper apiCall with credentials
export async function signup({ 
  fullName, 
  email, 
  password 
}: { 
  fullName: string; 
  email: string; 
  password: string;
}) {
  const cleanEmail = email.trim();
  const cleanPassword = password.trim();
  
  const baseUrl = getCurrentBaseUrl();
  
  try {
    console.log('[SIGNUP] Attempting signup...');
    const res = await fetch(`${baseUrl}/auth/signup`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      credentials: 'include', // ✅ Critical for CORS
      body: JSON.stringify({
        full_name: fullName.trim(),
        email: cleanEmail,
        password: cleanPassword,
      }),
    });
    
    if (!res.ok) {
      const error = await res.json().catch(() => ({}));
      throw new Error(error.detail || "Signup failed");
    }
    
    const data = await res.json();
    console.log('[SIGNUP] Success');
    
    if (data.access_token) {
      localStorage.setItem('access_token', data.access_token);
      localStorage.setItem('refresh_token', data.refresh_token);
      localStorage.setItem('user', JSON.stringify(data.user));
    }
    
    return data;
  } catch (err) {
    console.error('[SIGNUP] Error:', err);
    throw err;
  }
}

export async function login({ 
  email, 
  password 
}: { 
  email: string; 
  password: string;
}) {
  const cleanEmail = email.trim();
  const cleanPassword = password.trim();
  
  const baseUrl = getCurrentBaseUrl();
  
  try {
    console.log('[LOGIN] Attempting login...');
    const res = await fetch(`${baseUrl}/auth/login`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      credentials: 'include', // ✅ Critical for CORS
      body: JSON.stringify({
        email: cleanEmail,
        password: cleanPassword,
      }),
    });
    
    if (!res.ok) {
      const error = await res.json().catch(() => ({}));
      throw new Error(error.detail || "Login failed");
    }
    
    const data = await res.json();
    console.log('[LOGIN] Success');
    
    if (data.access_token) {
      localStorage.setItem('access_token', data.access_token);
      localStorage.setItem('refresh_token', data.refresh_token);
      localStorage.setItem('user', JSON.stringify(data.user));
    }
    
    return data;
  } catch (err) {
    console.error('[LOGIN] Error:', err);
    throw err;
  }
}

export async function uploadImage({ 
  file, 
  token 
}: { 
  file: File; 
  token: string;
}) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${getCurrentBaseUrl()}/upload/image`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
    },
    credentials: 'include',
    body: formData,
  });
  
  if (!res.ok) {
    const error = await res.json().catch(() => ({}));
    throw new Error(error.detail || "Image upload failed");
  }
  
  return res.json();
}

export async function runDiagnosis({
  imagePath,
  imageUrl,
  token
}: {
  imagePath?: string;
  imageUrl?: string;
  token: string;
}) {
  const res = await fetch(`${getCurrentBaseUrl()}/diagnosis/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    credentials: 'include',
    body: JSON.stringify({
      image_file_path: imagePath,
      image_url: imageUrl,
    }),
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({}));
    throw new Error(error.detail || "Diagnosis failed");
  }

  return res.json();
}

export async function runSegmentation({
  imagePath,
  imageUrl,
  token,
}: {
  imagePath?: string;
  imageUrl?: string;
  token: string;
}) {
  const res = await fetch(`${getCurrentBaseUrl()}/segmentation/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    credentials: "include",
    body: JSON.stringify({
      image_file_path: imagePath,
      image_url: imageUrl,
    }),
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({}));
    throw new Error(error.detail || "Segmentation failed");
  }

  return res.json();
}

export async function fetchExplainability({
  imageUrl,
  diagnosisData,
  token,
}: {
  imageUrl: string;
  diagnosisData: any;
  token?: string;
}) {
  return apiCall<{
    attention_heatmap_base64: string;
    gradcam_heatmap_base64: string;
    gpt_statement: string;
  }>("POST", "/explain", {
    image_url: imageUrl,
    diagnosis_data: diagnosisData,
  }, token);
}

// Token helpers
export function getToken(): string | null {
  return localStorage.getItem('access_token');
}

export function getRefreshToken(): string | null {
  return localStorage.getItem('refresh_token');
}

export function getCurrentUser() {
  const userStr = localStorage.getItem('user');
  return userStr ? JSON.parse(userStr) : null;
}

export function isLoggedIn(): boolean {
  return !!localStorage.getItem('access_token');
}

export function logout() {
  localStorage.removeItem('access_token');
  localStorage.removeItem('refresh_token');
  localStorage.removeItem('user');
  console.log('[LOGOUT] User logged out');
}

export async function clearAllHistory(token: string) {
  const res = await fetch(`${getCurrentBaseUrl()}/diagnosis/history`, {
    method: "DELETE",
    headers: {
      Authorization: `Bearer ${token}`,
    },
    credentials: "include",
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({}));
    throw new Error(error.detail || "Failed to clear history");
  }

  return res.json();
}
const form = document.getElementById('uploadForm');
const input = document.getElementById('imageInput');
const preview = document.getElementById('preview');
const previewWrap = document.getElementById('previewWrap');
const resultBox = document.getElementById('resultBox');
const recordsTable = document.getElementById('recordsTable');
const refreshBtn = document.getElementById('refreshBtn');
const logoutBtn = document.getElementById('logoutBtn');

// Use your specific Render API URL
const RENDER_API_URL = "https://cropintel-api.onrender.com";

// --- TOUCH KEYBOARD SUPPRESSION LOGIC ---
function bindKeyboardSuppression() {
  document.querySelectorAll('input[data-vk="true"], textarea[data-vk="true"]').forEach(field => {
    field.addEventListener('focus', (e) => {
      e.preventDefault();
      field.blur();
      if (typeof showKeyboard === 'function') {
        showKeyboard(field);
      }
    });
  });
}

bindKeyboardSuppression();

// --- MOBILE LOGOUT FIX ---
if (logoutBtn) {
    logoutBtn.onclick = async (e) => {
        if (e) e.preventDefault();
        
        try {
            localStorage.removeItem("demo_mode_session");
            sessionStorage.clear(); 
            
            if (typeof auth !== 'undefined') {
                const { signOut } = await import("https://www.gstatic.com/firebasejs/10.13.2/firebase-auth.js");
                await signOut(auth);
            }
            
            window.location.replace("/auth.html"); 
        } catch (error) {
            console.error("Logout Error:", error);
            window.location.href = "/auth.html";
        }
    };
}

// --- FRONTEND LOGIC ---
input.addEventListener('change', () => {
  const file = input.files[0];
  if (!file) return;
  preview.src = URL.createObjectURL(file);
  previewWrap.classList.remove('hidden');
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const file = input.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('image', file);
  
  const plantingDate = document.getElementById('plantingDateInput')?.value;
  if (plantingDate) formData.append('planting_date', plantingDate);

  resultBox.innerHTML = '<p style="color:#12345a;">Analyzing crop health... (May take 50s if server is sleeping)</p>';

  try {
    const res = await fetch(`${RENDER_API_URL}/api/analyze`, { 
      method: 'POST', 
      body: formData 
    });
    
    // Check if the response is valid JSON before parsing
    const contentType = res.headers.get("content-type");
    if (contentType && contentType.indexOf("application/json") !== -1) {
        const data = await res.json();

        if (!res.ok) {
          resultBox.innerHTML = `<p class="error">${data.error || 'Analysis failed.'}</p>`;
          return;
        }

        renderResult(data);
        loadRecords();
    } else {
        // Handle HTML error pages from Render
        const textError = await res.text();
        console.error("API returned non-JSON response:", textError);
        resultBox.innerHTML = `<p class="error">Server Error: The API returned an HTML page. Ensure your Render backend is fully live and checking logs for crashes.</p>`;
    }
  } catch (error) {
    resultBox.innerHTML = `<p class="error">Network error. Is the Render API live? ${error.message}</p>`;
  }
});

if (refreshBtn) {
  refreshBtn.addEventListener('click', loadRecords);
}

function renderResult(data) {
  resultBox.innerHTML = `
    <div class="metric-grid">
      <div class="metric"><strong>Crop</strong>${data.crop_name || 'N/A'}</div>
      <div class="metric"><strong>Confidence</strong>${data.confidence_percent || 0}%</div>
      <div class="metric"><strong>Health</strong><span class="status-${data.health_status}">${data.health_status}</span></div>
      <div class="metric"><strong>Health Score</strong>${data.health_score || 0}</div>
      <div class="metric"><strong>Yield Forecast</strong>${data.yield_forecast_kg || 0} kg</div>
      <div class="metric"><strong>Captured</strong>${data.captured_at || 'Just now'}</div>
    </div>
    <div class="section-box" style="margin-top:15px;">
      <h3>Analysis Notes</h3>
      <p>${data.possible_issue || 'No issues detected.'}</p>
    </div>
    <img src="${data.preview_url}" alt="Analyzed crop" style="width:100%;max-height:320px;object-fit:cover;border-radius:14px;border:1px solid #dce7f2;margin-top:15px;">
  `;
}

async function loadRecords() {
  if (!recordsTable) return;
  
  try {
    const res = await fetch(`${RENDER_API_URL}/api/health`);
    
    const contentType = res.headers.get("content-type");
    if (contentType && contentType.indexOf("application/json") !== -1) {
        const rows = await res.json();
        
        if (!rows || rows.length === 0) {
          recordsTable.innerHTML = '<p>No records found in database.</p>';
          return;
        }

        recordsTable.innerHTML = `
          <table style="width:100%; border-collapse: collapse;">
            <thead>
              <tr style="text-align:left; border-bottom: 1px solid #dce7f2;">
                <th style="padding:10px;">Date</th>
                <th style="padding:10px;">Crop</th>
                <th style="padding:10px;">Health</th>
                <th style="padding:10px;">Yield</th>
              </tr>
            </thead>
            <tbody>
              ${rows.map(row => `
                <tr style="border-bottom: 1px solid #f0f4f8;">
                  <td style="padding:10px;">${row.analysis_date || 'N/A'}</td>
                  <td style="padding:10px;">${row.crop_name}</td>
                  <td style="padding:10px;">${row.health_status}</td>
                  <td style="padding:10px;">${row.yield_forecast_kg} kg</td>
                </tr>
              `).join('')}
            </tbody>
          </table>
        `;
    }
  } catch (err) {
    recordsTable.innerHTML = '<p>Unable to fetch history from Render API.</p>';
  }
}

loadRecords();
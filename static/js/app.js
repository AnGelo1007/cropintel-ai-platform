const form = document.getElementById('uploadForm');
const input = document.getElementById('imageInput');
const preview = document.getElementById('preview');
const previewWrap = document.getElementById('previewWrap');
const resultBox = document.getElementById('resultBox');
const recordsTable = document.getElementById('recordsTable');
const refreshBtn = document.getElementById('refreshBtn');

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
  resultBox.innerHTML = 'Analyzing image...';

  const res = await fetch('/api/analyze', { method: 'POST', body: formData });
  const data = await res.json();

  if (!res.ok) {
    resultBox.innerHTML = `<p>${data.error || 'Upload failed.'}</p>`;
    return;
  }

  renderResult(data);
  loadRecords();
});

refreshBtn.addEventListener('click', loadRecords);

function renderResult(data) {
  resultBox.innerHTML = `
    <div class="metric-grid">
      <div class="metric"><strong>Crop</strong>${data.crop_name}</div>
      <div class="metric"><strong>Confidence</strong>${(data.crop_confidence * 100).toFixed(1)}%</div>
      <div class="metric"><strong>Health</strong><span class="status-${data.health_status}">${data.health_status}</span></div>
      <div class="metric"><strong>Health Score</strong>${data.health_score}</div>
      <div class="metric"><strong>Disease Risk</strong>${data.disease_risk}</div>
      <div class="metric"><strong>Yield Forecast</strong>${data.yield_forecast_kg} kg</div>
      <div class="metric"><strong>Model Loaded</strong>${data.model_loaded ? 'Yes' : 'Demo mode'}</div>
      <div class="metric"><strong>Cloud</strong>${data.cloud.message}</div>
    </div>
    <p><strong>Notes:</strong> ${data.notes}</p>
    <p><strong>Captured:</strong> ${data.captured_at}</p>
    <img src="${data.image_path}" alt="Analyzed crop" style="width:100%;max-height:320px;object-fit:cover;border-radius:14px;border:1px solid #dce7f2;">
  `;
}

async function loadRecords() {
  const res = await fetch('/api/records');
  const rows = await res.json();
  if (!rows.length) {
    recordsTable.innerHTML = '<p>No records yet.</p>';
    return;
  }
  recordsTable.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Date</th>
          <th>Crop</th>
          <th>Health</th>
          <th>Score</th>
          <th>Yield</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map(row => `
          <tr>
            <td>${row.id}</td>
            <td>${row.created_at}</td>
            <td>${row.crop_name}</td>
            <td>${row.health_status}</td>
            <td>${row.health_score}</td>
            <td>${row.yield_forecast_kg} kg</td>
          </tr>
        `).join('')}
      </tbody>
    </table>
  `;
}

loadRecords();

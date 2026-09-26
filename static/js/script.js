/**
 * Loan Default Prediction - Frontend Script
 * Handles Disclaimer Modal, Default Model Persistence, Form Submissions,
 * Real-time Predictions, Dataset Pagination, and Visual Comparisons.
 */

document.addEventListener('DOMContentLoaded', () => {
  initDisclaimerModal();
  initDefaultModel();
  initPredictionForm();
  initDatasetExplorer();
  initMobileNav();
});

/* ==========================================================================
   1. Disclaimer Modal
   ========================================================================== */
function initDisclaimerModal() {
  const modal = document.getElementById('disclaimer-modal');
  const acceptBtn = document.getElementById('accept-disclaimer-btn');
  const openLink = document.getElementById('open-disclaimer-link');

  if (!modal) return;

  const hasAccepted = localStorage.getItem('loan_ml_disclaimer_accepted');
  if (!hasAccepted) {
    modal.classList.add('active');
  }

  if (acceptBtn) {
    acceptBtn.addEventListener('click', () => {
      localStorage.setItem('loan_ml_disclaimer_accepted', 'true');
      modal.classList.remove('active');
    });
  }

  if (openLink) {
    openLink.addEventListener('click', (e) => {
      e.preventDefault();
      modal.classList.add('active');
    });
  }

  // Close on outside click
  modal.addEventListener('click', (e) => {
    if (e.target === modal && hasAccepted) {
      modal.classList.remove('active');
    }
  });
}

/* ==========================================================================
   2. Default Model Selection & LocalStorage Persistence
   ========================================================================== */
function getDefaultModel() {
  return localStorage.getItem('loan_ml_default_model') || 'Random Forest';
}

function setDefaultModel(modelName) {
  localStorage.setItem('loan_ml_default_model', modelName);
  updateDefaultModelUI(modelName);

  // Sync with backend API
  fetch('/api/default-model', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ default_model: modelName })
  }).catch(err => console.log('Backend sync notice:', err));
}

function updateDefaultModelUI(modelName) {
  // Update nav badge
  const navBadge = document.getElementById('nav-default-model-name');
  if (navBadge) navBadge.textContent = modelName;

  // Update homepage display
  const homeBadge = document.getElementById('home-default-model-name');
  if (homeBadge) homeBadge.textContent = modelName;

  // Update prediction form dropdown if not changed by user
  const modelSelect = document.getElementById('model-select');
  if (modelSelect && !modelSelect.dataset.userChanged) {
    for (let i = 0; i < modelSelect.options.length; i++) {
      if (modelSelect.options[i].value === modelName) {
        modelSelect.selectedIndex = i;
        break;
      }
    }
    const currentBadge = document.getElementById('selected-model-is-default-badge');
    if (currentBadge) {
      currentBadge.style.display = (modelSelect.value === modelName) ? 'inline-flex' : 'none';
    }
  }

  // Update comparison page default cards/radios
  const modelRadios = document.querySelectorAll('input[name="default_model_choice"]');
  modelRadios.forEach(radio => {
    if (radio.value === modelName) {
      radio.checked = true;
    }
  });

  const modelCards = document.querySelectorAll('.model-card-item');
  modelCards.forEach(card => {
    if (card.dataset.modelName === modelName) {
      card.classList.add('is-default');
    } else {
      card.classList.remove('is-default');
    }
  });
}

function initDefaultModel() {
  const currentDefault = getDefaultModel();
  updateDefaultModelUI(currentDefault);

  // If on comparison page, bind "Set as Default" button or radio changes
  const setDefaultBtn = document.getElementById('btn-set-default-model');
  if (setDefaultBtn) {
    setDefaultBtn.addEventListener('click', () => {
      const selected = document.querySelector('input[name="default_model_choice"]:checked');
      if (selected) {
        setDefaultModel(selected.value);
        showNotification(`Success: "${selected.value}" is now the Default Prediction Model!`, 'success');
      }
    });
  }

  // Direct card click to set default
  const cardSetBtns = document.querySelectorAll('.btn-make-default');
  cardSetBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
      const modelName = e.target.dataset.model;
      if (modelName) {
        setDefaultModel(modelName);
        showNotification(`Default Model updated to: "${modelName}"`, 'success');
      }
    });
  });
}

/* ==========================================================================
   3. Prediction Form
   ========================================================================== */
function initPredictionForm() {
  const form = document.getElementById('prediction-form');
  if (!form) return;

  const modelSelect = document.getElementById('model-select');
  const defaultBadge = document.getElementById('selected-model-is-default-badge');

  if (modelSelect) {
    // Select default model initially
    const currentDefault = getDefaultModel();
    for (let i = 0; i < modelSelect.options.length; i++) {
      if (modelSelect.options[i].value === currentDefault) {
        modelSelect.selectedIndex = i;
        break;
      }
    }
    if (defaultBadge) {
      defaultBadge.style.display = (modelSelect.value === currentDefault) ? 'inline-flex' : 'none';
    }

    modelSelect.addEventListener('change', () => {
      modelSelect.dataset.userChanged = 'true';
      if (defaultBadge) {
        defaultBadge.style.display = (modelSelect.value === getDefaultModel()) ? 'inline-flex' : 'none';
      }
    });
  }

  // Preset button listeners
  const btnLowRisk = document.getElementById('btn-preset-low-risk');
  const btnHighRisk = document.getElementById('btn-preset-high-risk');
  const btnReset = document.getElementById('btn-preset-reset');

  if (btnLowRisk) {
    btnLowRisk.addEventListener('click', () => loadSampleProfile({
      Age: 46,
      Income: 98000,
      LoanAmount: 22000,
      CreditScore: 780,
      MonthsEmployed: 60,
      NumCreditLines: 2,
      InterestRate: 5.5,
      LoanTerm: 36,
      DTIRatio: 0.22,
      Education: "Master's",
      EmploymentType: "Full-time",
      MaritalStatus: "Married",
      HasMortgage: "Yes",
      HasDependents: "No",
      LoanPurpose: "Home",
      HasCoSigner: "Yes"
    }));
  }

  if (btnHighRisk) {
    btnHighRisk.addEventListener('click', () => loadSampleProfile({
      Age: 21,
      Income: 18000,
      LoanAmount: 210000,
      CreditScore: 340,
      MonthsEmployed: 3,
      NumCreditLines: 4,
      InterestRate: 23.5,
      LoanTerm: 60,
      DTIRatio: 0.85,
      Education: "High School",
      EmploymentType: "Unemployed",
      MaritalStatus: "Single",
      HasMortgage: "No",
      HasDependents: "Yes",
      LoanPurpose: "Auto",
      HasCoSigner: "No"
    }));
  }

  if (btnReset) {
    btnReset.addEventListener('click', () => {
      form.reset();
      const currentDefault = getDefaultModel();
      if (modelSelect) modelSelect.value = currentDefault;
      const resultCard = document.getElementById('prediction-result-card');
      if (resultCard) resultCard.style.display = 'none';
    });
  }

  // Handle Form Submit
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const submitBtn = document.getElementById('btn-submit-predict');
    const originalBtnHtml = submitBtn.innerHTML;
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span class="spinner"></span> Processing Prediction...';

    const formData = new FormData(form);
    const payload = {};
    formData.forEach((value, key) => {
      payload[key] = value;
    });

    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Server prediction error');
      }

      displayPredictionResult(data);
    } catch (err) {
      showNotification(`Error: ${err.message}`, 'danger');
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = originalBtnHtml;
    }
  });
}

function loadSampleProfile(profile) {
  for (const [key, val] of Object.entries(profile)) {
    const input = document.getElementById(key);
    if (input) {
      input.value = val;
    }
  }
  showNotification('Sample data loaded successfully. Click "Predict Loan Default" to run model.', 'info');
}

function displayPredictionResult(res) {
  const card = document.getElementById('prediction-result-card');
  if (!card) return;

  const isDefault = res.prediction === 1;
  card.className = `result-card ${isDefault ? 'danger' : 'success'}`;
  card.style.display = 'block';

  document.getElementById('res-icon').textContent = isDefault ? '⚠️' : '✅';
  document.getElementById('res-title').textContent = isDefault ? 'Loan Default Predicted' : 'No Loan Default Predicted';
  document.getElementById('res-model-name').textContent = res.model_name;

  const defaultPct = res.default_probability.toFixed(2);
  const nonDefaultPct = (100 - res.default_probability).toFixed(2);
  const confidencePct = res.confidence_probability.toFixed(2);

  document.getElementById('res-default-prob').textContent = `${defaultPct}%`;
  document.getElementById('res-confidence-prob').textContent = `${confidencePct}%`;
  document.getElementById('res-risk-level').textContent = res.risk_level;

  // Gauge Fill
  const fill = document.getElementById('gauge-bar-fill');
  if (fill) {
    fill.style.width = `${Math.min(Math.max(res.default_probability, 3), 100)}%`;
    fill.style.background = isDefault 
      ? 'var(--gradient-danger)' 
      : 'var(--gradient-success)';
  }

  // Scroll to result card
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ==========================================================================
   4. Dataset Explorer & Server-Side Pagination
   ========================================================================== */
let datasetState = {
  page: 1,
  per_page: 50,
  search: '',
  default_filter: 'all'
};

function initDatasetExplorer() {
  const tableBody = document.getElementById('dataset-tbody');
  if (!tableBody) return;

  loadDatasetPage();

  const searchInput = document.getElementById('dataset-search');
  if (searchInput) {
    let debounceTimer;
    searchInput.addEventListener('input', (e) => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        datasetState.search = e.target.value.trim();
        datasetState.page = 1;
        loadDatasetPage();
      }, 350);
    });
  }

  const filterSelect = document.getElementById('dataset-filter-default');
  if (filterSelect) {
    filterSelect.addEventListener('change', (e) => {
      datasetState.default_filter = e.target.value;
      datasetState.page = 1;
      loadDatasetPage();
    });
  }

  const perPageSelect = document.getElementById('dataset-per-page');
  if (perPageSelect) {
    perPageSelect.addEventListener('change', (e) => {
      datasetState.per_page = parseInt(e.target.value, 10);
      datasetState.page = 1;
      loadDatasetPage();
    });
  }
}

async function loadDatasetPage() {
  const tableBody = document.getElementById('dataset-tbody');
  const infoSpan = document.getElementById('dataset-rows-info');
  const paginationControls = document.getElementById('dataset-pagination-controls');

  if (!tableBody) return;

  tableBody.innerHTML = `<tr><td colspan="18" style="text-align: center; padding: 2rem; color: var(--text-muted);"><span class="spinner"></span> Loading dataset records...</td></tr>`;

  try {
    const url = `/api/dataset?page=${datasetState.page}&per_page=${datasetState.per_page}&search=${encodeURIComponent(datasetState.search)}&default_filter=${encodeURIComponent(datasetState.default_filter)}`;
    const response = await fetch(url);
    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || 'Failed to load dataset');
    }

    renderDatasetRows(data.rows);
    if (infoSpan) {
      const start = (data.page - 1) * data.per_page + 1;
      const end = Math.min(data.page * data.per_page, data.total_rows);
      infoSpan.textContent = data.total_rows > 0 
        ? `Showing rows ${start.toLocaleString()}–${end.toLocaleString()} of ${data.total_rows.toLocaleString()}`
        : 'No records matching query';
    }
    renderPaginationButtons(data.page, data.total_pages);
  } catch (err) {
    tableBody.innerHTML = `<tr><td colspan="18" style="text-align: center; color: var(--status-danger); padding: 1.5rem;">Error: ${err.message}</td></tr>`;
  }
}

function renderDatasetRows(rows) {
  const tableBody = document.getElementById('dataset-tbody');
  if (!tableBody) return;

  if (rows.length === 0) {
    tableBody.innerHTML = `<tr><td colspan="18" style="text-align: center; padding: 2rem; color: var(--text-muted);">No records found matching criteria.</td></tr>`;
    return;
  }

  tableBody.innerHTML = rows.map(r => `
    <tr>
      <td><strong>${r.LoanID || '-'}</strong></td>
      <td>${r.Age}</td>
      <td>$${Number(r.Income).toLocaleString()}</td>
      <td>$${Number(r.LoanAmount).toLocaleString()}</td>
      <td>${r.CreditScore}</td>
      <td>${r.MonthsEmployed}m</td>
      <td>${r.NumCreditLines}</td>
      <td>${r.InterestRate}%</td>
      <td>${r.LoanTerm}m</td>
      <td>${Number(r.DTIRatio).toFixed(2)}</td>
      <td>${r.Education}</td>
      <td>${r.EmploymentType}</td>
      <td>${r.MaritalStatus}</td>
      <td>${r.HasMortgage}</td>
      <td>${r.HasDependents}</td>
      <td>${r.LoanPurpose}</td>
      <td>${r.HasCoSigner}</td>
      <td>
        <span class="badge ${r.Default === 1 ? 'badge-danger' : 'badge-success'}">
          ${r.Default === 1 ? 'Default (1)' : 'No Default (0)'}
        </span>
      </td>
    </tr>
  `).join('');
}

function renderPaginationButtons(currentPage, totalPages) {
  const container = document.getElementById('dataset-pagination-controls');
  if (!container) return;

  if (totalPages <= 1) {
    container.innerHTML = '';
    return;
  }

  let html = `
    <button class="page-btn" ${currentPage === 1 ? 'disabled' : ''} onclick="changePage(1)">«</button>
    <button class="page-btn" ${currentPage === 1 ? 'disabled' : ''} onclick="changePage(${currentPage - 1})">‹</button>
  `;

  const windowSize = 2;
  const startPage = Math.max(1, currentPage - windowSize);
  const endPage = Math.min(totalPages, currentPage + windowSize);

  if (startPage > 1) {
    html += `<button class="page-btn" onclick="changePage(1)">1</button>`;
    if (startPage > 2) html += `<span style="padding: 0 4px; color: var(--text-muted);">...</span>`;
  }

  for (let i = startPage; i <= endPage; i++) {
    html += `<button class="page-btn ${i === currentPage ? 'active' : ''}" onclick="changePage(${i})">${i}</button>`;
  }

  if (endPage < totalPages) {
    if (endPage < totalPages - 1) html += `<span style="padding: 0 4px; color: var(--text-muted);">...</span>`;
    html += `<button class="page-btn" onclick="changePage(${totalPages})">${totalPages}</button>`;
  }

  html += `
    <button class="page-btn" ${currentPage === totalPages ? 'disabled' : ''} onclick="changePage(${currentPage + 1})">›</button>
    <button class="page-btn" ${currentPage === totalPages ? 'disabled' : ''} onclick="changePage(${totalPages})">»</button>
  `;

  container.innerHTML = html;
}

window.changePage = function(page) {
  datasetState.page = page;
  loadDatasetPage();
  window.scrollTo({ top: document.querySelector('.table-responsive').offsetTop - 80, behavior: 'smooth' });
};

/* ==========================================================================
   5. Notification Toast
   ========================================================================== */
function showNotification(msg, type = 'info') {
  let toast = document.getElementById('toast-notification');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'toast-notification';
    toast.style.cssText = `
      position: fixed;
      bottom: 24px;
      right: 24px;
      padding: 0.9rem 1.4rem;
      border-radius: 8px;
      color: #fff;
      font-size: 0.9rem;
      font-weight: 600;
      z-index: 9999;
      box-shadow: 0 8px 24px rgba(0,0,0,0.5);
      transition: all 0.3s ease;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    `;
    document.body.appendChild(toast);
  }

  const bgColors = {
    success: 'rgba(16, 185, 129, 0.95)',
    danger: 'rgba(239, 68, 68, 0.95)',
    info: 'rgba(59, 130, 246, 0.95)'
  };

  toast.style.background = bgColors[type] || bgColors.info;
  toast.textContent = msg;
  toast.style.opacity = '1';
  toast.style.transform = 'translateY(0)';

  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(12px)';
  }, 3800);
}

/* ==========================================================================
   6. Mobile Nav
   ========================================================================== */
function initMobileNav() {
  const toggleBtn = document.getElementById('nav-toggle-btn');
  const navMenu = document.getElementById('nav-menu');
  if (toggleBtn && navMenu) {
    toggleBtn.addEventListener('click', () => {
      navMenu.classList.toggle('show');
    });
  }
}

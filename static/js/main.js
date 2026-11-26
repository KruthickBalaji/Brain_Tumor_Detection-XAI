// main.js - Brain Tumor Detection & Explainability Web Platform

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Lucide icons
  if (window.lucide) {
    window.lucide.createIcons();
  }

  // --- Element References ---
  const themeToggleBtn = document.getElementById('themeToggleBtn');
  const themeIcon = document.getElementById('themeIcon');
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const uploadPrompt = document.getElementById('uploadPrompt');
  const previewContainer = document.getElementById('previewContainer');
  const imagePreview = document.getElementById('imagePreview');
  const fileInfoBadge = document.getElementById('fileInfoBadge');
  const removeImageBtn = document.getElementById('removeImageBtn');
  const analyzeBtn = document.getElementById('analyzeBtn');

  const emptyState = document.getElementById('emptyState');
  const loadingState = document.getElementById('loadingState');
  const resultsContainer = document.getElementById('resultsContainer');

  const resDiagnosisLabel = document.getElementById('resDiagnosisLabel');
  const resBadgeStatus = document.getElementById('resBadgeStatus');
  const untrainedWarningBanner = document.getElementById('untrainedWarningBanner');
  const resConfidence = document.getElementById('resConfidence');
  const resConfidenceBar = document.getElementById('resConfidenceBar');
  const resOriginalImage = document.getElementById('resOriginalImage');
  const resHeatmapImage = document.getElementById('resHeatmapImage');
  const xaiCardWrapper = document.getElementById('xaiCardWrapper');
  const xaiHeaderTitle = document.getElementById('xaiHeaderTitle');
  const xaiHeaderBadge = document.getElementById('xaiHeaderBadge');
  const camLegendContainer = document.getElementById('camLegendContainer');
  const probabilitiesContainer = document.getElementById('probabilitiesContainer');
  const clinicalNoteText = document.getElementById('clinicalNoteText');
  const sampleButtonsContainer = document.getElementById('sampleButtonsContainer');

  let currentFile = null;

  // --- Theme Management ---
  const savedTheme = localStorage.getItem('theme') || 'dark';
  applyTheme(savedTheme);

  themeToggleBtn.addEventListener('click', () => {
    const isDark = document.documentElement.classList.contains('dark');
    applyTheme(isDark ? 'light' : 'dark');
  });

  function applyTheme(theme) {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
      themeIcon.setAttribute('data-lucide', 'sun');
    } else {
      document.documentElement.classList.remove('dark');
      themeIcon.setAttribute('data-lucide', 'moon');
    }
    localStorage.setItem('theme', theme);
    if (window.lucide) window.lucide.createIcons();
  }

  // --- File Drag & Drop Handling ---
  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('border-cyan-500', 'bg-slate-900/90');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('border-cyan-500', 'bg-slate-900/90');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files.length > 0) {
      handleFileSelection(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelection(e.target.files[0]);
    }
  });

  removeImageBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
  });

  function handleFileSelection(file) {
    if (!file.type.match('image.*')) {
      alert('Please upload a valid image file (.png, .jpg, .jpeg)');
      return;
    }
    currentFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      fileInfoBadge.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      uploadPrompt.classList.add('hidden');
      previewContainer.classList.remove('hidden');
      analyzeBtn.disabled = false;
      if (window.lucide) window.lucide.createIcons();
    };
    reader.readAsDataURL(file);
  }

  function resetUpload() {
    currentFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    fileInfoBadge.textContent = '';
    previewContainer.classList.add('hidden');
    uploadPrompt.classList.remove('hidden');
    analyzeBtn.disabled = true;
  }

  // --- Quick Test Sample Cases ---
  if (sampleButtonsContainer) {
    sampleButtonsContainer.addEventListener('click', async (e) => {
      const btn = e.target.closest('.sample-btn');
      if (!btn) return;
      const sampleType = btn.dataset.sample;
      try {
        btn.classList.add('opacity-50');
        const resp = await fetch(`/api/sample/${sampleType}`);
        if (!resp.ok) {
          throw new Error('Failed to load sample image');
        }
        const blob = await resp.blob();
        const file = new File([blob], `sample_${sampleType}.jpg`, { type: 'image/jpeg' });
        handleFileSelection(file);
      } catch (err) {
        console.error(err);
        alert('Could not load sample MRI scan: ' + err.message);
      } finally {
        btn.classList.remove('opacity-50');
      }
    });
  }

  // --- Clinical Descriptions Map ---
  const clinicalInterpretations = {
    'glioma_tumor': {
      title: 'Glioma Detected',
      badgeClass: 'bg-red-500/20 text-red-400 border-red-500/30',
      note: 'The model has identified high-confidence focal activations corresponding to intra-axial glial neoplasia. Grad-CAM saliency highlights irregular, infiltrative hyperintense boundaries characteristic of high-grade astrocytoma or glioblastoma. Recommend neuro-oncology referral and stereotactic biopsy evaluation.'
    },
    'meningioma_tumor': {
      title: 'Meningioma Detected',
      badgeClass: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
      note: 'The network detected extra-axial dural-based focal hyperintensity with hallmark dural tail sign activations. Grad-CAM confirms localized focus along the meningeal periphery without deep parenchymal infiltration. Typically benign; surgical resection or stereotactic radiosurgery consultation advised.'
    },
    'pituitary_tumor': {
      title: 'Pituitary Tumor Detected',
      badgeClass: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
      note: 'Saliency concentrates strictly within the sellar and suprasellar compartments, indicating macroadenoma or pituitary endocrine expansion. Grad-CAM highlights potential mass-effect proximity to the optic chiasm. Recommend endocrine hormone panel and ophthalmologic visual field perimetry.'
    },
    'no_tumor': {
      title: 'No Tumor Identified',
      badgeClass: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
      note: 'Neuroimaging exhibits preserved bilateral symmetry, unremarkable ventricular margins, and lack of focal mass effect or hyperintense signal anomalies. Grad-CAM shows diffuse, non-pathological background distribution consistent with normal neuroanatomy.'
    }
  };

  // Human-readable labels
  const classDisplayName = {
    'glioma_tumor': 'Glioma',
    'meningioma_tumor': 'Meningioma',
    'no_tumor': 'No Tumor',
    'pituitary_tumor': 'Pituitary'
  };

  const classColors = {
    'glioma_tumor': 'from-red-500 to-rose-600',
    'meningioma_tumor': 'from-amber-500 to-orange-600',
    'no_tumor': 'from-emerald-500 to-teal-600',
    'pituitary_tumor': 'from-purple-500 to-indigo-600'
  };

  // --- Analyze Prediction CTA ---
  analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    // Switch UI to Loading State
    emptyState.classList.add('hidden');
    resultsContainer.classList.add('hidden');
    loadingState.classList.remove('hidden');
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = `
      <svg class="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
      </svg>
      <span>Computing Backbone Feature Maps & Grad-CAM...</span>
    `;

    try {
      const formData = new FormData();
      formData.append('file', currentFile);

      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server responded with status ${response.status}`);
      }

      const data = await response.json();
      renderResults(data);

    } catch (err) {
      console.error('Inference error:', err);
      alert('Diagnostic analysis error: ' + err.message);
      loadingState.classList.add('hidden');
      emptyState.classList.remove('hidden');
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.innerHTML = `
        <i data-lucide="sparkles" class="w-4 h-4"></i>
        <span>Run Diagnostic & Grad-CAM Analysis</span>
      `;
      if (window.lucide) window.lucide.createIcons();
    }
  });

  // --- Render Results ---
  function renderResults(data) {
    loadingState.classList.add('hidden');
    resultsContainer.classList.remove('hidden');

    if (untrainedWarningBanner) {
      if (data.model_trained === false) {
        untrainedWarningBanner.classList.remove('hidden');
      } else {
        untrainedWarningBanner.classList.add('hidden');
      }
    }

    const predKey = data.prediction;
    const confVal = typeof data.confidence === 'number' ? data.confidence : parseFloat(data.confidence);
    const meta = clinicalInterpretations[predKey] || {
      title: predKey,
      badgeClass: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
      note: 'Diagnostic inference completed.'
    };

    // 1. Diagnosis Header & Badge
    resDiagnosisLabel.textContent = meta.title;
    resBadgeStatus.textContent = classDisplayName[predKey] || predKey;
    resBadgeStatus.className = `px-3 py-1 rounded-full text-xs font-bold border uppercase ${meta.badgeClass}`;

    // 2. Confidence percentage & bar
    resConfidence.textContent = `${confVal.toFixed(1)}%`;
    resConfidenceBar.style.width = `${confVal.toFixed(1)}%`;

    // 3. Side-by-side images
    resOriginalImage.src = data.original_image;
    resHeatmapImage.src = data.heatmap_image;

    // Update XAI card header, badge, and legend based on tumor status
    const hasTumor = data.has_tumor !== undefined ? data.has_tumor : (predKey !== 'no_tumor');
    if (!hasTumor) {
      if (xaiHeaderTitle) {
        xaiHeaderTitle.className = 'text-xs font-semibold text-emerald-400 flex items-center gap-1.5';
        xaiHeaderTitle.innerHTML = '<i data-lucide="shield-check" class="w-4 h-4 text-emerald-400"></i><span>Anatomical Verification (Negative)</span>';
      }
      if (xaiHeaderBadge) {
        xaiHeaderBadge.className = 'text-[11px] px-2 py-0.5 rounded bg-emerald-950/60 text-emerald-300 border border-emerald-500/30 font-medium';
        xaiHeaderBadge.textContent = 'No Tumor Lesion';
      }
      if (camLegendContainer) {
        camLegendContainer.innerHTML = '<i data-lucide="check-circle-2" class="w-3.5 h-3.5 text-emerald-400"></i><span class="text-emerald-400 font-medium">Clear / No Focal Pathology</span>';
      }
      if (xaiCardWrapper) {
        xaiCardWrapper.classList.remove('border-cyan-500/40', 'border-amber-500/40');
        xaiCardWrapper.classList.add('border-emerald-500/40');
      }
    } else {
      if (xaiHeaderTitle) {
        xaiHeaderTitle.className = 'text-xs font-semibold text-cyan-300 flex items-center gap-1.5';
        xaiHeaderTitle.innerHTML = '<i data-lucide="flame" class="w-4 h-4 text-amber-400"></i><span>Focal Tumor Activation</span>';
      }
      if (xaiHeaderBadge) {
        xaiHeaderBadge.className = 'text-[11px] px-2 py-0.5 rounded bg-cyan-950/60 text-cyan-300 border border-cyan-500/30 font-medium';
        xaiHeaderBadge.textContent = 'Tumor Localized';
      }
      if (camLegendContainer) {
        camLegendContainer.innerHTML = '<span>Activation:</span><div class="w-12 h-2 rounded bg-gradient-to-r from-blue-600 via-green-400 via-yellow-400 to-red-600"></div><span class="text-red-400 font-bold">High</span>';
      }
      if (xaiCardWrapper) {
        xaiCardWrapper.classList.remove('border-emerald-500/40');
        xaiCardWrapper.classList.add('border-cyan-500/40');
      }
    }

    // 4. Clinical notes
    clinicalNoteText.textContent = meta.note;

    // 5. Probability distribution bars
    renderProbabilities(data.probabilities, predKey);

    // Reinitialize icons in newly revealed DOM elements
    if (window.lucide) window.lucide.createIcons();

    // Smooth scroll down to results on mobile devices
    if (window.innerWidth < 1024) {
      resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  function renderProbabilities(probabilities, winningClass) {
    probabilitiesContainer.innerHTML = '';

    // Sort classes: winning class first or standard order
    const classKeys = ['glioma_tumor', 'meningioma_tumor', 'pituitary_tumor', 'no_tumor'];

    classKeys.forEach((key) => {
      const prob = (probabilities && probabilities[key] !== undefined) ? probabilities[key] : 0;
      const percentage = (prob * 100).toFixed(1);
      const isWinner = key === winningClass;
      const displayName = classDisplayName[key] || key;
      const gradientClass = classColors[key] || 'from-cyan-500 to-blue-600';

      const row = document.createElement('div');
      row.className = 'space-y-1';
      row.innerHTML = `
        <div class="flex items-center justify-between text-xs font-medium">
          <span class="flex items-center gap-1.5 ${isWinner ? 'text-white font-bold' : 'text-slate-400'}">
            ${isWinner ? '<i data-lucide="check" class="w-3.5 h-3.5 text-cyan-400"></i>' : '<span class="w-3.5 inline-block"></span>'}
            ${displayName}
          </span>
          <span class="font-mono ${isWinner ? 'text-cyan-400 font-bold' : 'text-slate-400'}">${percentage}%</span>
        </div>
        <div class="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-700/60">
          <div class="h-full bg-gradient-to-r ${gradientClass} rounded-full transition-all duration-700" style="width: ${percentage}%"></div>
        </div>
      `;
      probabilitiesContainer.appendChild(row);
    });
  }
});

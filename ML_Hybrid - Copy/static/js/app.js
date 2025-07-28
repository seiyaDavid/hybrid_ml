// Global variables
let currentAnalysis = null;
let selectedFile = null;
let csvColumns = [];
let selectedColumn = null;

// DOM elements
const loadingOverlay = document.getElementById('loading-overlay');
const fileInput = document.getElementById('file-input');
const uploadZone = document.getElementById('upload-zone');
const uploadBtn = document.getElementById('upload-btn');
const navBtns = document.querySelectorAll('.nav-btn');
const tabContents = document.querySelectorAll('.tab-content');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const chatMessages = document.getElementById('chat-messages');
const quickActionBtns = document.querySelectorAll('.quick-action-btn');
const reportItems = document.querySelectorAll('.report-item');
const themeModal = document.getElementById('theme-modal');
const modalClose = document.getElementById('modal-close');
const columnSelection = document.getElementById('column-selection');
const summaryColumnSelect = document.getElementById('summary-column-select');

// Initialize the application
document.addEventListener('DOMContentLoaded', function () {
    initializeEventListeners();
    showTab('upload');
});

// Initialize all event listeners
function initializeEventListeners() {
    // Navigation
    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.getAttribute('data-tab');
            showTab(tab);
        });
    });

    // File upload
    uploadZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    uploadBtn.addEventListener('click', handleFileUpload);

    // Drag and drop
    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('dragleave', handleDragLeave);
    uploadZone.addEventListener('drop', handleDrop);

    // Column selection
    summaryColumnSelect.addEventListener('change', handleColumnSelection);

    // Chat
    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    // Quick actions
    quickActionBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const question = btn.getAttribute('data-question');
            chatInput.value = question;
            sendMessage();
        });
    });

    // Reports
    reportItems.forEach(item => {
        item.addEventListener('click', () => {
            const reportType = item.getAttribute('data-report');
            showReport(reportType);
        });
    });

    // Modal
    modalClose.addEventListener('click', closeModal);
    themeModal.addEventListener('click', (e) => {
        if (e.target === themeModal) closeModal();
    });

    // Quality filter
    const qualityFilter = document.getElementById('quality-filter');
    if (qualityFilter) {
        qualityFilter.addEventListener('change', filterThemes);
    }
}

// Tab navigation
function showTab(tabName) {
    // Update navigation buttons
    navBtns.forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('data-tab') === tabName) {
            btn.classList.add('active');
        }
    });

    // Show selected tab content
    tabContents.forEach(content => {
        content.classList.remove('active');
        if (content.id === `${tabName}-tab`) {
            content.classList.add('active');
        }
    });
}

// File handling
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        selectedFile = file;
        updateUploadZone(file.name);
        previewCSVColumns(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    uploadZone.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadZone.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    uploadZone.classList.remove('dragover');

    const files = event.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
            selectedFile = file;
            fileInput.files = files;
            updateUploadZone(file.name);
            previewCSVColumns(file);
        } else {
            showNotification('Please select a CSV file', 'error');
        }
    }
}

function updateUploadZone(fileName) {
    uploadZone.innerHTML = `
        <i class="fas fa-file-csv"></i>
        <p>${fileName}</p>
        <span class="file-info">Ready to upload</span>
    `;
}

// CSV Column Preview
async function previewCSVColumns(file) {
    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/preview-columns', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            csvColumns = result.columns;
            populateColumnDropdown();
            columnSelection.classList.remove('hidden');
            uploadBtn.disabled = true; // Disable until column is selected
        } else {
            showNotification(result.error || 'Failed to preview columns', 'error');
        }
    } catch (error) {
        console.error('Column preview error:', error);
        showNotification('Failed to preview CSV columns', 'error');
    }
}

function populateColumnDropdown() {
    summaryColumnSelect.innerHTML = '<option value="">Select a column...</option>';

    csvColumns.forEach(column => {
        const option = document.createElement('option');
        option.value = column;
        option.textContent = column;
        summaryColumnSelect.appendChild(option);
    });
}

function handleColumnSelection() {
    selectedColumn = summaryColumnSelect.value;
    if (selectedColumn) {
        uploadBtn.disabled = false;
        showNotification(`Selected column: ${selectedColumn}`, 'success');
    } else {
        uploadBtn.disabled = true;
    }
}

// File upload and analysis
async function handleFileUpload() {
    if (!selectedFile) {
        showNotification('Please select a file first', 'error');
        return;
    }

    if (!selectedColumn) {
        showNotification('Please select a summary column first', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('summary_column', selectedColumn);

    showLoading(true);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            currentAnalysis = result.analysis;
            showNotification('Analysis completed successfully!', 'success');
            updateAnalysisSummary(result.analysis);
            showTab('themes');
            displayThemes(result.analysis.theme_analysis);
            displayClusterStats(result.analysis.clustering_results);
        } else {
            showNotification(result.error || 'Analysis failed', 'error');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showNotification('Upload failed. Please try again.', 'error');
    } finally {
        showLoading(false);
    }
}

// Analysis summary
function updateAnalysisSummary(analysis) {
    const summary = document.getElementById('analysis-summary');
    if (!summary) return;

    document.getElementById('total-records').textContent = analysis.data_info?.total_records || '-';
    document.getElementById('themes-found').textContent = Object.keys(analysis.theme_analysis || {}).length;
    document.getElementById('clusters-found').textContent = analysis.clustering_results?.statistics?.num_clusters || '-';
    document.getElementById('quality-issues').textContent = analysis.data_quality_analysis?.data_quality_count || '-';

    summary.classList.remove('hidden');
}

// Themes display
function displayThemes(themes) {
    const themesGrid = document.getElementById('themes-grid');
    if (!themesGrid || !themes) return;

    themesGrid.innerHTML = '';

    Object.entries(themes).forEach(([id, theme]) => {
        const themeCard = createThemeCard(id, theme);
        themesGrid.appendChild(themeCard);
    });
}

function createThemeCard(id, theme) {
    const card = document.createElement('div');
    card.className = 'theme-card';
    card.addEventListener('click', () => showThemeDetails(id, theme));

    const qualityIcon = theme.data_quality_issues > 0 ?
        '<i class="fas fa-exclamation-triangle" style="color: #ef4444;"></i>' :
        '<i class="fas fa-check-circle" style="color: #10b981;"></i>';

    card.innerHTML = `
        <div class="theme-header">
            <div>
                <div class="theme-title">${theme.name}</div>
                <div class="theme-cluster">Cluster ${id}</div>
            </div>
            ${qualityIcon}
        </div>
        <div class="theme-description">${theme.description}</div>
        <div class="theme-stats">
            <div class="theme-stat">
                <div class="theme-stat-value">${theme.sample_count}</div>
                <div class="theme-stat-label">Samples</div>
            </div>
            <div class="theme-stat">
                <div class="theme-stat-value">${theme.data_quality_issues}</div>
                <div class="theme-stat-label">Quality Issues</div>
            </div>
        </div>
    `;

    return card;
}

function showThemeDetails(id, theme) {
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');

    modalTitle.textContent = theme.name;
    modalBody.innerHTML = `
        <div style="margin-bottom: 1.5rem;">
            <h4 style="color: white; margin-bottom: 0.5rem;">Description</h4>
            <p style="color: rgba(255, 255, 255, 0.8); line-height: 1.6;">${theme.description}</p>
        </div>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #8b5cf6;">${theme.sample_count}</div>
                <div style="font-size: 0.8rem; color: rgba(255, 255, 255, 0.6);">Samples</div>
            </div>
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #ef4444;">${theme.data_quality_issues}</div>
                <div style="font-size: 0.8rem; color: rgba(255, 255, 255, 0.6);">Quality Issues</div>
            </div>
            <div style="background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 10px; text-align: center;">
                <div style="font-size: 1.5rem; font-weight: bold; color: #10b981;">${theme.data_quality_percentage?.toFixed(1) || 0}%</div>
                <div style="font-size: 0.8rem; color: rgba(255, 255, 255, 0.6);">Issue Rate</div>
            </div>
        </div>
        ${theme.representative_examples ? `
            <div>
                <h4 style="color: white; margin-bottom: 0.5rem;">Representative Examples</h4>
                <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                    ${theme.representative_examples.slice(0, 3).map(example =>
        `<div style="background: rgba(255, 255, 255, 0.1); padding: 0.75rem; border-radius: 8px; font-size: 0.9rem; color: rgba(255, 255, 255, 0.8);">${example}</div>`
    ).join('')}
                </div>
            </div>
        ` : ''}
    `;

    themeModal.classList.remove('hidden');
}

function closeModal() {
    themeModal.classList.add('hidden');
}

// Cluster statistics
function displayClusterStats(clusteringResults) {
    if (!clusteringResults?.statistics) return;

    const stats = clusteringResults.statistics;
    document.getElementById('total-points').textContent = stats.total_points || '-';
    document.getElementById('num-clusters').textContent = stats.num_clusters || '-';
    document.getElementById('noise-points').textContent = stats.num_noise_points || '-';
    document.getElementById('silhouette-score').textContent = (stats.silhouette_score || 0).toFixed(3);
}

// Chat functionality
async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    // Add user message
    addMessage(message, true);
    chatInput.value = '';

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                context: { analysis_id: currentAnalysis?.analysis_id }
            })
        });

        const result = await response.json();

        if (result.success) {
            addMessage(result.message, false);
        } else {
            addMessage('Sorry, I encountered an error. Please try again.', false);
        }
    } catch (error) {
        console.error('Chat error:', error);
        addMessage('Sorry, I encountered an error. Please try again.', false);
    }
}

function addMessage(text, isUser) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;

    const icon = isUser ? 'fas fa-user' : 'fas fa-robot';
    const iconColor = isUser ? 'white' : '#8b5cf6';

    messageDiv.innerHTML = `
        <div class="message-content">
            <i class="${icon}" style="color: ${iconColor};"></i>
            <p>${text}</p>
        </div>
    `;

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Reports
function showReport(reportType) {
    // Update active report item
    reportItems.forEach(item => {
        item.classList.remove('active');
        if (item.getAttribute('data-report') === reportType) {
            item.classList.add('active');
        }
    });

    // Show report content (placeholder)
    const reportViewer = document.querySelector('.report-viewer');
    if (reportViewer) {
        reportViewer.innerHTML = `
            <div style="text-align: center; padding: 2rem;">
                <i class="fas fa-file-alt" style="font-size: 3rem; color: rgba(255, 255, 255, 0.3); margin-bottom: 1rem;"></i>
                <h3 style="color: white; margin-bottom: 0.5rem;">${reportType.replace('_', ' ').toUpperCase()}</h3>
                <p style="color: rgba(255, 255, 255, 0.6);">Report content will be displayed here</p>
            </div>
        `;
    }
}

// Theme filtering
function filterThemes() {
    const qualityFilter = document.getElementById('quality-filter');
    const themesGrid = document.getElementById('themes-grid');

    if (!qualityFilter || !themesGrid || !currentAnalysis?.theme_analysis) return;

    const showOnlyQualityIssues = qualityFilter.checked;
    const themeCards = themesGrid.querySelectorAll('.theme-card');

    themeCards.forEach(card => {
        const qualityIssues = parseInt(card.querySelector('.theme-stat-value').textContent);
        card.style.display = showOnlyQualityIssues && qualityIssues === 0 ? 'none' : 'block';
    });
}

// Loading overlay
function showLoading(show) {
    if (show) {
        loadingOverlay.classList.remove('hidden');
    } else {
        loadingOverlay.classList.add('hidden');
    }
}

// Notifications
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        color: white;
        font-weight: 500;
        z-index: 10000;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transform: translateX(100%);
        transition: transform 0.3s ease;
    `;

    // Set background color based on type
    const colors = {
        success: 'linear-gradient(135deg, #10b981, #059669)',
        error: 'linear-gradient(135deg, #ef4444, #dc2626)',
        info: 'linear-gradient(135deg, #3b82f6, #2563eb)'
    };

    notification.style.background = colors[type] || colors.info;
    notification.textContent = message;

    document.body.appendChild(notification);

    // Animate in
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);

    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Utility functions
function formatNumber(num) {
    return new Intl.NumberFormat().format(num);
}

function formatPercentage(num) {
    return `${(num * 100).toFixed(1)}%`;
}

// Export functions for global access
window.showTab = showTab;
window.showNotification = showNotification; 
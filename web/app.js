/**
 * LegoGen Web App
 * See plan.md for full specification
 */

// Configuration
const API_URL = 'http://localhost:8000';

// Brick dimensions: [width, length]
const BRICK_DIMS = {
    "1x1": [1, 1],
    "1x2": [1, 2],
    "1x3": [1, 3],
    "1x4": [1, 4],
    "1x6": [1, 6],
    "2x2": [2, 2],
    "2x3": [2, 3],
    "2x4": [2, 4],
    "2x6": [2, 6],
};

// LEGO colors for brick rendering
const BRICK_COLORS = [
    '#D01012', // Red
    '#0055BF', // Blue
    '#237841', // Green
    '#F5CD2F', // Yellow
    '#FFA500', // Orange
    '#69007F', // Purple
    '#00A3DA', // Light Blue
];

// Application state
let state = {
    imageFile: null,
    description: '',
    loading: false,
    error: null,
    result: null,
    currentLayer: 0,
};

// DOM Elements
const imageUpload = document.getElementById('image-upload');
const uploadZone = document.getElementById('upload-zone');
const imagePreview = document.getElementById('image-preview');
const uploadText = document.querySelector('.upload-text');
const uploadIcon = document.querySelector('.upload-icon');
const descriptionInput = document.getElementById('description');
const charCurrent = document.getElementById('char-current');
const submitBtn = document.getElementById('submit-btn');
const btnText = document.querySelector('.btn-text');
const btnLoading = document.querySelector('.btn-loading');
const errorSection = document.getElementById('error-section');
const errorMessage = document.getElementById('error-message');
const resultsSection = document.getElementById('results-section');
const metaDimensions = document.getElementById('meta-dimensions');
const metaBricks = document.getElementById('meta-bricks');
const prevLayerBtn = document.getElementById('prev-layer');
const nextLayerBtn = document.getElementById('next-layer');
const layerIndicator = document.getElementById('layer-indicator');
const layerCanvas = document.getElementById('layer-canvas');
const ctx = layerCanvas.getContext('2d');

// Initialize event listeners
function init() {
    // Image upload
    imageUpload.addEventListener('change', handleImageSelect);

    // Drag and drop
    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('dragleave', handleDragLeave);
    uploadZone.addEventListener('drop', handleDrop);

    // Description input
    descriptionInput.addEventListener('input', handleDescriptionInput);

    // Submit button
    submitBtn.addEventListener('click', handleSubmit);

    // Layer navigation
    prevLayerBtn.addEventListener('click', () => navigateLayer(-1));
    nextLayerBtn.addEventListener('click', () => navigateLayer(1));
}

// Image handling
function handleImageSelect(e) {
    const file = e.target.files[0];
    if (file) {
        setImage(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    uploadZone.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadZone.classList.remove('dragover');

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        setImage(file);
    }
}

function setImage(file) {
    state.imageFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.style.display = 'block';
        uploadText.style.display = 'none';
        uploadIcon.style.display = 'none';
    };
    reader.readAsDataURL(file);

    updateSubmitButton();
}

// Description handling
function handleDescriptionInput(e) {
    state.description = e.target.value;
    charCurrent.textContent = state.description.length;
    updateSubmitButton();
}

// Form validation
function updateSubmitButton() {
    const isValid = state.imageFile && state.description.trim().length > 0;
    submitBtn.disabled = !isValid || state.loading;
}

// Submit handler
async function handleSubmit() {
    if (state.loading) return;

    state.loading = true;
    state.error = null;
    state.result = null;

    // Update UI
    submitBtn.classList.add('loading');
    btnText.style.display = 'none';
    btnLoading.style.display = 'inline';
    submitBtn.disabled = true;
    errorSection.style.display = 'none';
    resultsSection.style.display = 'none';

    try {
        const formData = new FormData();
        formData.append('image', state.imageFile);
        formData.append('description', state.description);

        const response = await fetch(`${API_URL}/build`, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (data.success) {
            state.result = data;
            state.currentLayer = 0;
            showResults();
        } else {
            showError(data.error || 'Failed to generate build instructions');
        }
    } catch (err) {
        showError(`Connection error: ${err.message}`);
    } finally {
        state.loading = false;
        submitBtn.classList.remove('loading');
        btnText.style.display = 'inline';
        btnLoading.style.display = 'none';
        updateSubmitButton();
    }
}

// Error display
function showError(message) {
    state.error = message;
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
}

// Results display
function showResults() {
    const { layers, metadata } = state.result;

    // Update metadata
    if (metadata.voxel_dimensions) {
        const [w, d, h] = metadata.voxel_dimensions;
        metaDimensions.textContent = `Size: ${w} × ${d} × ${h}`;
    }
    if (metadata.total_bricks) {
        metaBricks.textContent = `Total bricks: ${metadata.total_bricks}`;
    }

    // Show results section
    resultsSection.style.display = 'block';

    // Render first layer
    updateLayerDisplay();
}

// Layer navigation
function navigateLayer(delta) {
    const newLayer = state.currentLayer + delta;
    const maxLayer = state.result.layers.length - 1;

    if (newLayer >= 0 && newLayer <= maxLayer) {
        state.currentLayer = newLayer;
        updateLayerDisplay();
    }
}

function updateLayerDisplay() {
    const { layers, metadata } = state.result;
    const totalLayers = layers.length;

    // Update indicator
    layerIndicator.textContent = `Layer ${state.currentLayer + 1} of ${totalLayers}`;

    // Update button states
    prevLayerBtn.disabled = state.currentLayer === 0;
    nextLayerBtn.disabled = state.currentLayer === totalLayers - 1;

    // Render the layer
    const layer = layers[state.currentLayer];
    const [gridWidth, gridHeight] = metadata.voxel_dimensions || [16, 16];
    renderLayer(layer, gridWidth, gridHeight);
}

// Layer rendering
function renderLayer(layer, gridWidth, gridHeight) {
    // Calculate cell size to fit canvas
    const canvasSize = 400;
    const maxDim = Math.max(gridWidth, gridHeight);
    const cellSize = Math.floor(canvasSize / maxDim);

    // Resize canvas to fit grid
    layerCanvas.width = gridWidth * cellSize;
    layerCanvas.height = gridHeight * cellSize;

    // Clear canvas
    ctx.fillStyle = '#e8e8e8';
    ctx.fillRect(0, 0, layerCanvas.width, layerCanvas.height);

    // Draw grid lines
    ctx.strokeStyle = '#ccc';
    ctx.lineWidth = 1;

    for (let x = 0; x <= gridWidth; x++) {
        ctx.beginPath();
        ctx.moveTo(x * cellSize, 0);
        ctx.lineTo(x * cellSize, gridHeight * cellSize);
        ctx.stroke();
    }

    for (let y = 0; y <= gridHeight; y++) {
        ctx.beginPath();
        ctx.moveTo(0, y * cellSize);
        ctx.lineTo(gridWidth * cellSize, y * cellSize);
        ctx.stroke();
    }

    // Draw bricks
    layer.forEach((brick, index) => {
        const coverage = getBrickCoverage(brick);
        const color = BRICK_COLORS[index % BRICK_COLORS.length];

        const x = brick.x * cellSize;
        const y = brick.y * cellSize;
        const w = coverage.w * cellSize;
        const h = coverage.h * cellSize;

        // Fill brick
        ctx.fillStyle = color;
        ctx.fillRect(x + 2, y + 2, w - 4, h - 4);

        // Brick border
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 2, y + 2, w - 4, h - 4);

        // Draw studs
        drawStuds(x, y, coverage.w, coverage.h, cellSize);
    });
}

function getBrickCoverage(brick) {
    const dims = BRICK_DIMS[brick.type];
    if (!dims) {
        console.warn(`Unknown brick type: ${brick.type}`);
        return { w: 1, h: 1 };
    }

    const [width, length] = dims;

    if (brick.rotation === 0) {
        // length along X, width along Y
        return { w: length, h: width };
    } else {
        // length along Y, width along X
        return { w: width, h: length };
    }
}

function drawStuds(brickX, brickY, widthCells, heightCells, cellSize) {
    const studRadius = cellSize * 0.25;

    ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';

    for (let dy = 0; dy < heightCells; dy++) {
        for (let dx = 0; dx < widthCells; dx++) {
            const cx = brickX + (dx + 0.5) * cellSize;
            const cy = brickY + (dy + 0.5) * cellSize;

            ctx.beginPath();
            ctx.arc(cx, cy, studRadius, 0, Math.PI * 2);
            ctx.fill();
        }
    }
}

// Initialize on load
init();

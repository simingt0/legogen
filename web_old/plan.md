# Web Frontend Module

## Overview
Svelte-based web application for uploading brick images, entering descriptions, and viewing generated LEGO build instructions layer by layer.

## Architecture
```
┌─────────────────────────────────────────────────────────────┐
│  Web App (localhost:8080)                                   │
│  ├── Image upload component                                 │
│  ├── Description input                                      │
│  ├── Submit button → POST /build to server                  │
│  └── Results display                                        │
│      └── Layer-by-layer 2D grid renderer                    │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
    FastAPI Server (localhost:8000)
```

## Tech Stack
- **Svelte** (or vanilla JS for simplicity given time constraints)
- **No build step for MVP**: Use Svelte via CDN or plain HTML/CSS/JS
- **Styling**: LEGO-themed (bright colors, chunky elements, playful)

## Pages / Views

### Main View (single page app)
1. **Header**: "LegoGen" title with LEGO styling
2. **Input Section**:
   - Image upload (drag & drop or click to browse)
   - Text input for description
   - Submit button
3. **Loading State**: Show while waiting for server
4. **Results Section**:
   - Layer navigation (prev/next or slider)
   - 2D grid showing current layer
   - Layer number indicator "Layer 3 of 8"

## API Integration

### POST /build
```javascript
async function submitBuild(imageFile, description) {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('description', description);

    const response = await fetch('http://localhost:8000/build', {
        method: 'POST',
        body: formData,
    });

    return await response.json();
}
```

### Response Handling
```javascript
// Success response
{
    "success": true,
    "layers": [
        [
            {"type": "2x4", "x": 0, "y": 0, "rotation": 0},
            ...
        ],
        ...
    ],
    "metadata": {
        "voxel_dimensions": [8, 6, 4],
        "total_bricks": 15
    }
}

// Error response
{
    "success": false,
    "error": "Failed to generate build after 5 attempts"
}
```

## Layer Renderer

### Input
A single layer: array of brick placements
```javascript
[
    {"type": "2x4", "x": 0, "y": 0, "rotation": 0},
    {"type": "2x2", "x": 4, "y": 0, "rotation": 0},
]
```

### Output
2D grid visualization using HTML Canvas or SVG.

### Rendering Logic
```javascript
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

function getBrickCoverage(brick) {
    const [width, length] = BRICK_DIMS[brick.type];
    if (brick.rotation === 0) {
        // length along X, width along Y
        return { w: length, h: width };
    } else {
        // length along Y, width along X
        return { w: width, h: length };
    }
}

function renderLayer(ctx, layer, gridWidth, gridHeight, cellSize) {
    // Draw grid background
    ctx.fillStyle = '#e0e0e0';
    ctx.fillRect(0, 0, gridWidth * cellSize, gridHeight * cellSize);

    // Draw grid lines
    ctx.strokeStyle = '#ccc';
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
    const colors = ['#D01012', '#0055BF', '#237841', '#F5CD2F', '#FFA500'];
    layer.forEach((brick, i) => {
        const { w, h } = getBrickCoverage(brick);
        const color = colors[i % colors.length];

        ctx.fillStyle = color;
        ctx.fillRect(
            brick.x * cellSize + 1,
            brick.y * cellSize + 1,
            w * cellSize - 2,
            h * cellSize - 2
        );

        // Brick outline
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 2;
        ctx.strokeRect(
            brick.x * cellSize + 1,
            brick.y * cellSize + 1,
            w * cellSize - 2,
            h * cellSize - 2
        );

        // Optional: draw studs
        // ...
    });
}
```

## LEGO Styling Guide

### Colors
```css
:root {
    --lego-red: #D01012;
    --lego-blue: #0055BF;
    --lego-yellow: #F5CD2F;
    --lego-green: #237841;
    --lego-orange: #FFA500;
    --lego-black: #1B1B1B;
    --lego-white: #FFFFFF;
    --lego-gray: #8B8B8B;
}
```

### Typography
- Bold, playful fonts (e.g., system-ui with heavy weight)
- Large, readable text

### UI Elements
- Rounded corners (8-12px)
- Drop shadows for depth
- Bright primary colors for buttons
- Chunky, tactile-feeling elements

## File Structure
```
web/
├── plan.md
├── index.html      # Main HTML file
├── styles.css      # LEGO-themed styles
└── app.js          # Application logic
```

## Running the Frontend
```bash
cd web
python -m http.server 8080
```

Then open http://localhost:8080 in a browser.

## Component Breakdown

### 1. ImageUpload
- Drag & drop zone
- File input fallback
- Image preview after selection
- Accepts: JPEG, PNG

### 2. DescriptionInput
- Textarea with placeholder "Describe what you want to build..."
- Character counter (max 600)

### 3. SubmitButton
- Disabled until image and description provided
- Loading state while request in progress

### 4. LayerViewer
- Canvas element for rendering
- Prev/Next buttons for navigation
- Layer counter display
- Grid size adapts to voxel_dimensions from response

### 5. ErrorDisplay
- Shows error message when success=false
- Styled to be noticeable but not alarming

## State Management
```javascript
let state = {
    // Inputs
    imageFile: null,
    description: '',

    // Request state
    loading: false,
    error: null,

    // Results
    result: null,
    currentLayer: 0,
};
```

## Test Cases

### Test 1: Image upload
- Select image file
- Verify preview shows
- Verify file is stored in state

### Test 2: Form validation
- Empty description → submit disabled
- No image → submit disabled
- Both provided → submit enabled

### Test 3: API call
- Submit form
- Verify loading state shown
- Verify request sent to correct endpoint

### Test 4: Success rendering
- Mock successful response
- Verify layers render correctly
- Verify navigation works

### Test 5: Error handling
- Mock error response
- Verify error message displays

## Accessibility Notes
- Use proper form labels
- Keyboard navigation for layer controls
- Alt text for any images
- Sufficient color contrast

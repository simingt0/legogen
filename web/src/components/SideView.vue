<template>
  <div class="side-view">
    <div class="view-container" ref="viewContainer">
      <canvas ref="canvas" @click="handleCanvasClick"></canvas>
    </div>

    <div class="rotation-controls">
      <button class="rotate-btn" @click="rotateLeft" title="Rotate Left">
        <div class="brick-arrow left">
          <div class="arrow-row"><span class="empty"></span><span class="empty"></span><span class="brick"></span></div>
          <div class="arrow-row"><span class="empty"></span><span class="brick"></span><span class="brick"></span></div>
          <div class="arrow-row"><span class="brick"></span><span class="brick"></span><span class="brick"></span></div>
          <div class="arrow-row"><span class="empty"></span><span class="brick"></span><span class="brick"></span></div>
          <div class="arrow-row"><span class="empty"></span><span class="empty"></span><span class="brick"></span></div>
        </div>
      </button>
      <button class="rotate-btn" @click="rotateRight" title="Rotate Right">
        <div class="brick-arrow right">
          <div class="arrow-row"><span class="brick"></span><span class="empty"></span><span class="empty"></span></div>
          <div class="arrow-row"><span class="brick"></span><span class="brick"></span><span class="empty"></span></div>
          <div class="arrow-row"><span class="brick"></span><span class="brick"></span><span class="brick"></span></div>
          <div class="arrow-row"><span class="brick"></span><span class="brick"></span><span class="empty"></span></div>
          <div class="arrow-row"><span class="brick"></span><span class="empty"></span><span class="empty"></span></div>
        </div>
      </button>
    </div>

    <div class="view-mode-hint">
      <span class="hint-text">Press SPACE to toggle view</span>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { store } from '../store'

const props = defineProps({
  currentLayer: {
    type: Number,
    default: 0
  },
  viewAngle: {
    type: Number,
    default: 0
  }
})

const emit = defineEmits(['update:currentLayer', 'viewAngleChange', 'update:isCornerView'])

const canvas = ref(null)
const viewContainer = ref(null)
const isCornerView = ref(false)
let ctx = null
let resizeObserver = null
let layerHitboxes = []

const BRICK_DIMS = {
  '1x1': [1, 1], '1x2': [1, 2], '1x3': [1, 3],
  '1x4': [1, 4], '1x6': [1, 6], '2x2': [2, 2],
  '2x3': [2, 3], '2x4': [2, 4], '2x6': [2, 6]
}

const BRICK_COLORS = [
  '#DA291C', // red
  '#0055BF', // blue
  '#237841', // green
  '#FFA500', // orange
  '#FFD700', // yellow
  '#800080', // purple
  '#00CED1'  // cyan
]

const layers = computed(() => store.buildResult?.layers || [])
const totalLayers = computed(() => layers.value.length)

const gridSize = computed(() => {
  const dims = store.buildResult?.metadata?.voxel_dimensions
  return dims ? { width: dims[0], depth: dims[1], height: dims[2] } : { width: 10, depth: 10, height: 5 }
})

function getVisibleBricksFromAngle(layer, angle) {
  // Returns only the front-most visible bricks from the given viewing angle
  // angle: 0=front(+Y), 1=right(+X), 2=back(-Y), 3=left(-X)

  const grid = gridSize.value
  const frontmost = new Map() // key: position along viewing axis, value: {depth, brickIndex}

  layer.forEach((brick, brickIndex) => {
    const dims = BRICK_DIMS[brick.type] || [1, 1]
    const brickWidth = dims[0]
    const brickLength = dims[1]

    let w, h
    if (brick.rotation === 90) {
      w = brickWidth
      h = brickLength
    } else {
      w = brickLength
      h = brickWidth
    }

    // Mark all cells this brick occupies
    for (let dx = 0; dx < w; dx++) {
      for (let dy = 0; dy < h; dy++) {
        const cellX = brick.x + dx
        const cellY = brick.y + dy

        let viewPos, depth

        switch(angle) {
          case 0: // Front view (+Y direction)
            viewPos = grid.width - 1 - cellX
            depth = grid.depth - 1 - cellY
            break
          case 1: // Right view (+X direction)
            viewPos = cellY
            depth = grid.width - 1 - cellX
            break
          case 2: // Back view (-Y direction)
            viewPos = cellX
            depth = cellY
            break
          case 3: // Left view (-X direction)
            viewPos = grid.width - 1 - cellY
            depth = cellX
            break
        }

        const key = viewPos
        if (!frontmost.has(key) || frontmost.get(key).depth < depth) {
          frontmost.set(key, { depth, brickIndex })
        }
      }
    }
  })

  return frontmost
}

function drawView() {
  if (isCornerView.value) {
    drawCornerView()
  } else {
    drawSideView()
  }
}

function drawSideView() {
  if (!canvas.value || !ctx || !viewContainer.value) return

  const canvasEl = canvas.value
  const containerWidth = viewContainer.value.clientWidth
  const containerHeight = viewContainer.value.clientHeight

  canvasEl.width = containerWidth
  canvasEl.height = containerHeight

  const width = canvasEl.width
  const height = canvasEl.height

  // Clear with light background
  ctx.fillStyle = '#f5f5f5'
  ctx.fillRect(0, 0, width, height)

  if (totalLayers.value === 0) {
    ctx.fillStyle = '#999'
    ctx.font = '14px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('No layers', width / 2, height / 2)
    return
  }

  const padding = 30
  const availableWidth = width - padding * 2
  const availableHeight = height - padding * 2

  const grid = gridSize.value

  // Make grid square - use the smaller dimension
  const maxCellSize = Math.floor(Math.min(
    availableWidth / grid.width,
    availableHeight / totalLayers.value
  ))

  const cellSize = Math.max(6, maxCellSize * 0.8) // Shrink a bit

  const totalWidth = grid.width * cellSize
  const totalHeight = totalLayers.value * cellSize

  const startX = (width - totalWidth) / 2
  const startY = (height - totalHeight) / 2

  // Draw each layer from bottom to top
  for (let layerIdx = 0; layerIdx < totalLayers.value; layerIdx++) {
    const layer = layers.value[layerIdx]
    const isCurrent = layerIdx === props.currentLayer

    // Calculate Y position (bottom layer at bottom)
    const y = startY + (totalLayers.value - 1 - layerIdx) * cellSize

    // Draw occupied cells
    const visibleBricks = getVisibleBricksFromAngle(layer, props.viewAngle)

    if (visibleBricks.size > 0) {
      // Draw blocks for visible columns
      for (const [pos, data] of visibleBricks.entries()) {
        const blockX = startX + parseInt(pos) * cellSize
        const blockWidth = cellSize - 1
        const blockHeight = cellSize - 1

        // Color - only for current layer, show actual brick colors
        if (isCurrent) {
          ctx.fillStyle = BRICK_COLORS[data.brickIndex % BRICK_COLORS.length]
        } else {
          // Gray for non-current layers
          ctx.fillStyle = '#d0d0d0'
        }

        ctx.fillRect(blockX, y, blockWidth, blockHeight)

        // Add border
        ctx.strokeStyle = isCurrent ? 'rgba(0, 0, 0, 0.3)' : 'rgba(0, 0, 0, 0.15)'
        ctx.lineWidth = isCurrent ? 1 : 0.5
        ctx.strokeRect(blockX, y, blockWidth, blockHeight)
      }
    }
  }
}

function handleCanvasClick(event) {
  if (isCornerView.value) {
    // In corner view, clicking does nothing
    return
  }

  // Side view click-to-select functionality
  if (!canvas.value) return

  const rect = canvas.value.getBoundingClientRect()
  const x = event.clientX - rect.left
  const y = event.clientY - rect.top

  const grid = gridSize.value
  const padding = 30
  const availableWidth = canvas.value.width - padding * 2
  const availableHeight = canvas.value.height - padding * 2

  const cellSize = Math.max(6, Math.min(
    availableWidth / grid.width,
    availableHeight / totalLayers.value
  ) * 0.8)

  const totalWidth = grid.width * cellSize
  const totalHeight = totalLayers.value * cellSize
  const startX = (canvas.value.width - totalWidth) / 2
  const startY = (canvas.value.height - totalHeight) / 2

  // Check which layer was clicked
  for (let layerIdx = 0; layerIdx < totalLayers.value; layerIdx++) {
    const y_pos = startY + (totalLayers.value - 1 - layerIdx) * cellSize

    if (x >= startX && x <= startX + totalWidth &&
        y >= y_pos && y <= y_pos + cellSize) {
      emit('update:currentLayer', layerIdx)
      break
    }
  }
}



function drawCornerView() {
  if (!canvas.value || !ctx || !viewContainer.value) return

  const canvasEl = canvas.value
  const containerWidth = viewContainer.value.clientWidth
  const containerHeight = viewContainer.value.clientHeight

  canvasEl.width = containerWidth
  canvasEl.height = containerHeight

  const width = canvasEl.width
  const height = canvasEl.height

  // Clear with light background
  ctx.fillStyle = '#f5f5f5'
  ctx.fillRect(0, 0, width, height)

  if (totalLayers.value === 0) {
    ctx.fillStyle = '#999'
    ctx.font = '14px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText('No layers', width / 2, height / 2)
    return
  }

  const grid = gridSize.value
  const padding = 20

  // Isometric projection constants - zoomed in
  const cellSize = Math.min(
    (width - padding * 2) / (grid.width + grid.depth),
    (height - padding * 2) / (totalLayers.value + grid.width + grid.depth)
  ) * 1.2

  const layerHeight = cellSize // Make cubic - same height as width/depth

  // Calculate center point - offset to center the grid at [0,0,0] in world space
  const gridCenterX = (grid.width - 1) / 2
  const gridCenterY = (grid.depth - 1) / 2
  const gridCenterZ = (totalLayers.value - 1) / 2

  const centerX = width / 2
  const centerY = height / 2

  // Isometric basis vectors (adjusted based on view angle)
  let isoX, isoY, isoZ
  switch(props.viewAngle) {
    case 0: // Front - looking from +Y corner (bottom right)
      isoX = { x: cellSize * 0.866, y: cellSize * 0.5 }
      isoY = { x: -cellSize * 0.866, y: cellSize * 0.5 }
      break
    case 1: // Right - looking from +X corner (top right)
      isoX = { x: -cellSize * 0.866, y: cellSize * 0.5 }
      isoY = { x: -cellSize * 0.866, y: -cellSize * 0.5 }
      break
    case 2: // Back - looking from -Y corner (top left)
      isoX = { x: -cellSize * 0.866, y: -cellSize * 0.5 }
      isoY = { x: cellSize * 0.866, y: -cellSize * 0.5 }
      break
    case 3: // Left - looking from -X corner (bottom left)
      isoX = { x: cellSize * 0.866, y: -cellSize * 0.5 }
      isoY = { x: cellSize * 0.866, y: cellSize * 0.5 }
      break
  }
  isoZ = { x: 0, y: -layerHeight }

  // Project 3D point to 2D isometric (centered at grid center)
  function project(x, y, z) {
    const worldX = x - gridCenterX
    const worldY = y - gridCenterY
    const worldZ = z - gridCenterZ
    return {
      x: centerX + isoX.x * worldX + isoY.x * worldY + isoZ.x * worldZ,
      y: centerY + isoX.y * worldX + isoY.y * worldY + isoZ.y * worldZ
    }
  }

  // Create a 3D grid to track which voxels are filled
  const filledVoxels = new Set()
  const voxelData = new Map()

  for (let layerIdx = 0; layerIdx < totalLayers.value; layerIdx++) {
    // Skip layers above the current layer
    if (layerIdx > props.currentLayer) {
      continue
    }

    const layer = layers.value[layerIdx]
    const isCurrent = layerIdx === props.currentLayer

    for (const brick of layer) {
      const dims = BRICK_DIMS[brick.type] || [1, 1]
      const brickWidth = dims[0]
      const brickLength = dims[1]

      let w, h
      if (brick.rotation === 90) {
        w = brickWidth
        h = brickLength
      } else {
        w = brickLength
        h = brickWidth
      }

      // Mark voxels for this brick
      for (let dx = 0; dx < w; dx++) {
        for (let dy = 0; dy < h; dy++) {
          const key = `${brick.x + dx},${brick.y + dy},${layerIdx}`
          filledVoxels.add(key)
          voxelData.set(key, {
            x: brick.x + dx,
            y: brick.y + dy,
            z: layerIdx,
            isCurrent,
            brickIndex: layer.indexOf(brick)
          })
        }
      }
    }
  }

  // Generate all grid voxels (including empty ones) - only up to current layer
  const allVoxels = []
  for (let z = 0; z <= props.currentLayer; z++) {
    for (let x = 0; x < grid.width; x++) {
      for (let y = 0; y < grid.depth; y++) {
        const key = `${x},${y},${z}`
        const isFilled = filledVoxels.has(key)

        if (isFilled) {
          allVoxels.push(voxelData.get(key))
        } else {
          allVoxels.push({
            x, y, z,
            isCurrent: false,
            isEmpty: true
          })
        }
      }
    }
  }

  // Sort voxels by depth (painter's algorithm)
  // Current layer should always be drawn last (on top)
  allVoxels.sort((a, b) => {
    // Current layer always drawn last
    if (a.isCurrent && !b.isCurrent) return 1
    if (!a.isCurrent && b.isCurrent) return -1

    // For same layer priority, sort by depth
    const depthA = a.x * isoX.y + a.y * isoY.y + a.z * isoZ.y
    const depthB = b.x * isoX.y + b.y * isoY.y + b.z * isoZ.y
    return depthA - depthB
  })

  // Draw in multiple passes to ensure proper layering
  // Pass 1: Non-current layers
  for (const voxel of allVoxels) {
    if (voxel.isEmpty || voxel.isCurrent) continue

    const p1 = project(voxel.x, voxel.y, voxel.z)
    const p2 = project(voxel.x + 1, voxel.y, voxel.z)
    const p3 = project(voxel.x + 1, voxel.y + 1, voxel.z)
    const p5 = project(voxel.x, voxel.y, voxel.z + 1)
    const p6 = project(voxel.x + 1, voxel.y, voxel.z + 1)
    const p7 = project(voxel.x + 1, voxel.y + 1, voxel.z + 1)
    const p8 = project(voxel.x, voxel.y + 1, voxel.z + 1)

    // Other layers - transparent gray
    ctx.fillStyle = '#d0d0d0'
    ctx.globalAlpha = 0.3

    // Draw top face
    ctx.beginPath()
    ctx.moveTo(p5.x, p5.y)
    ctx.lineTo(p6.x, p6.y)
    ctx.lineTo(p7.x, p7.y)
    ctx.lineTo(p8.x, p8.y)
    ctx.closePath()
    ctx.fill()

    // Draw front face
    ctx.fillStyle = '#b0b0b0'
    ctx.globalAlpha = 0.2
    ctx.beginPath()
    ctx.moveTo(p1.x, p1.y)
    ctx.lineTo(p2.x, p2.y)
    ctx.lineTo(p6.x, p6.y)
    ctx.lineTo(p5.x, p5.y)
    ctx.closePath()
    ctx.fill()

    // Draw side face
    ctx.fillStyle = '#909090'
    ctx.globalAlpha = 0.15
    ctx.beginPath()
    ctx.moveTo(p2.x, p2.y)
    ctx.lineTo(p3.x, p3.y)
    ctx.lineTo(p7.x, p7.y)
    ctx.lineTo(p6.x, p6.y)
    ctx.closePath()
    ctx.fill()

    // Draw edges
    ctx.globalAlpha = 1.0
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)'
    ctx.lineWidth = 0.5

    ctx.beginPath()
    ctx.moveTo(p5.x, p5.y)
    ctx.lineTo(p6.x, p6.y)
    ctx.lineTo(p7.x, p7.y)
    ctx.lineTo(p8.x, p8.y)
    ctx.closePath()
    ctx.stroke()
  }

  // Pass 2: Current layer side faces
  for (const voxel of allVoxels) {
    if (!voxel.isCurrent) continue

    const p1 = project(voxel.x, voxel.y, voxel.z)
    const p2 = project(voxel.x + 1, voxel.y, voxel.z)
    const p3 = project(voxel.x + 1, voxel.y + 1, voxel.z)
    const p4 = project(voxel.x, voxel.y + 1, voxel.z)
    const p5 = project(voxel.x, voxel.y, voxel.z + 1)
    const p6 = project(voxel.x + 1, voxel.y, voxel.z + 1)
    const p7 = project(voxel.x + 1, voxel.y + 1, voxel.z + 1)
    const p8 = project(voxel.x, voxel.y + 1, voxel.z + 1)

    // Current layer - show in fully opaque solid color
    ctx.fillStyle = BRICK_COLORS[voxel.brickIndex % BRICK_COLORS.length]
    ctx.globalAlpha = 1.0

    // Draw all 4 side faces - fully opaque

    // Front face (along X axis, min Y)
    ctx.beginPath()
    ctx.moveTo(p1.x, p1.y)
    ctx.lineTo(p2.x, p2.y)
    ctx.lineTo(p6.x, p6.y)
    ctx.lineTo(p5.x, p5.y)
    ctx.closePath()
    ctx.fill()

    // Right face (along Y axis, max X)
    ctx.beginPath()
    ctx.moveTo(p2.x, p2.y)
    ctx.lineTo(p3.x, p3.y)
    ctx.lineTo(p7.x, p7.y)
    ctx.lineTo(p6.x, p6.y)
    ctx.closePath()
    ctx.fill()

    // Back face (along X axis, max Y)
    ctx.beginPath()
    ctx.moveTo(p3.x, p3.y)
    ctx.lineTo(p4.x, p4.y)
    ctx.lineTo(p8.x, p8.y)
    ctx.lineTo(p7.x, p7.y)
    ctx.closePath()
    ctx.fill()

    // Left face (along Y axis, min X)
    ctx.beginPath()
    ctx.moveTo(p4.x, p4.y)
    ctx.lineTo(p1.x, p1.y)
    ctx.lineTo(p5.x, p5.y)
    ctx.lineTo(p8.x, p8.y)
    ctx.closePath()
    ctx.fill()
  }

  // Pass 3: Current layer top faces (drawn last, on top of everything)
  for (const voxel of allVoxels) {
    if (!voxel.isCurrent) continue

    const p5 = project(voxel.x, voxel.y, voxel.z + 1)
    const p6 = project(voxel.x + 1, voxel.y, voxel.z + 1)
    const p7 = project(voxel.x + 1, voxel.y + 1, voxel.z + 1)
    const p8 = project(voxel.x, voxel.y + 1, voxel.z + 1)

    // Current layer - show in fully opaque solid color
    ctx.fillStyle = BRICK_COLORS[voxel.brickIndex % BRICK_COLORS.length]
    ctx.globalAlpha = 1.0

    // Draw top face
    ctx.beginPath()
    ctx.moveTo(p5.x, p5.y)
    ctx.lineTo(p6.x, p6.y)
    ctx.lineTo(p7.x, p7.y)
    ctx.lineTo(p8.x, p8.y)
    ctx.closePath()
    ctx.fill()

    // Draw edges
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.4)'
    ctx.lineWidth = 1

    ctx.beginPath()
    ctx.moveTo(p5.x, p5.y)
    ctx.lineTo(p6.x, p6.y)
    ctx.lineTo(p7.x, p7.y)
    ctx.lineTo(p8.x, p8.y)
    ctx.closePath()
    ctx.stroke()
  }

  ctx.globalAlpha = 1.0

  // Draw bounding box edges for the entire grid dimensions
  ctx.strokeStyle = 'rgba(0, 0, 0, 0.3)'
  ctx.lineWidth = 2

  // Define the 8 corners of the bounding box
  const corners = [
    project(0, 0, 0),
    project(grid.width, 0, 0),
    project(grid.width, grid.depth, 0),
    project(0, grid.depth, 0),
    project(0, 0, totalLayers.value),
    project(grid.width, 0, totalLayers.value),
    project(grid.width, grid.depth, totalLayers.value),
    project(0, grid.depth, totalLayers.value),
  ]

  // Draw bottom face
  ctx.beginPath()
  ctx.moveTo(corners[0].x, corners[0].y)
  ctx.lineTo(corners[1].x, corners[1].y)
  ctx.lineTo(corners[2].x, corners[2].y)
  ctx.lineTo(corners[3].x, corners[3].y)
  ctx.closePath()
  ctx.stroke()

  // Draw top face
  ctx.beginPath()
  ctx.moveTo(corners[4].x, corners[4].y)
  ctx.lineTo(corners[5].x, corners[5].y)
  ctx.lineTo(corners[6].x, corners[6].y)
  ctx.lineTo(corners[7].x, corners[7].y)
  ctx.closePath()
  ctx.stroke()

  // Draw vertical edges
  ctx.beginPath()
  ctx.moveTo(corners[0].x, corners[0].y)
  ctx.lineTo(corners[4].x, corners[4].y)
  ctx.stroke()

  ctx.beginPath()
  ctx.moveTo(corners[1].x, corners[1].y)
  ctx.lineTo(corners[5].x, corners[5].y)
  ctx.stroke()

  ctx.beginPath()
  ctx.moveTo(corners[2].x, corners[2].y)
  ctx.lineTo(corners[6].x, corners[6].y)
  ctx.stroke()

  ctx.beginPath()
  ctx.moveTo(corners[3].x, corners[3].y)
  ctx.lineTo(corners[7].x, corners[7].y)
  ctx.stroke()
}

function toggleView() {
  isCornerView.value = !isCornerView.value
  emit('update:isCornerView', isCornerView.value)
  drawView()
}

function handleKeydown(e) {
  if (e.code === 'Space') {
    e.preventDefault()
    toggleView()
  }
}

function rotateLeft() {
  const newAngle = (props.viewAngle + 3) % 4
  emit('viewAngleChange', newAngle)
}

function rotateRight() {
  const newAngle = (props.viewAngle + 1) % 4
  emit('viewAngleChange', newAngle)
}

watch(() => props.currentLayer, drawView)
watch(() => props.viewAngle, drawView)
watch(() => store.buildResult, drawView, { deep: true })

onMounted(() => {
  ctx = canvas.value.getContext('2d')

  resizeObserver = new ResizeObserver(() => {
    drawView()
  })
  resizeObserver.observe(viewContainer.value)

  window.addEventListener('keydown', handleKeydown)
  drawView()
})

onUnmounted(() => {
  if (resizeObserver) {
    resizeObserver.disconnect()
  }
  window.removeEventListener('keydown', handleKeydown)
})
</script>

<style scoped>
.side-view {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.view-container {
  flex: 1;
  min-height: 0;
  background: var(--lego-white);
  border: 2px solid var(--lego-black);
  border-radius: var(--radius-md);
  position: relative;
  overflow: hidden;
}

canvas {
  width: 100%;
  height: 100%;
  display: block;
  cursor: pointer;
}

.rotation-controls {
  display: flex;
  gap: var(--spacing-sm);
  margin-top: var(--spacing-sm);
}

.rotate-btn {
  flex: 1;
  background: var(--lego-white);
  border: 2px solid var(--lego-black);
  border-radius: var(--radius-md);
  padding: 6px;
  cursor: pointer;
  transition: background 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.rotate-btn:hover {
  background: #f0f0f0;
}

.rotate-btn:active {
  background: #e0e0e0;
}

.brick-arrow {
  display: flex;
  flex-direction: column;
  gap: 1.5px;
  transform: scale(0.75);
  align-items: center;
  justify-content: center;
}

.arrow-row {
  display: flex;
  gap: 1.5px;
}

.brick-arrow .brick {
  width: 10px;
  height: 10px;
  background: var(--lego-red);
  border-radius: 2px;
  position: relative;
}

.brick-arrow .brick::after {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 5px;
  height: 5px;
  background: #b8231a;
  border-radius: 50%;
}

.brick-arrow .empty {
  width: 10px;
  height: 10px;
}

.view-mode-hint {
  text-align: center;
  margin-top: var(--spacing-xs);
}

.hint-text {
  font-size: 11px;
  color: #999;
  font-style: italic;
}
</style>

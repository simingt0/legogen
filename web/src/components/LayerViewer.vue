<template>
  <div class="layer-viewer">
    <div class="header">
      <span class="layer-indicator">Layer {{ currentLayer + 1 }} of {{ totalLayers }}</span>
      <button class="nav-btn arrow-btn" :disabled="currentLayer >= totalLayers - 1" @click="nextLayer">
        <div class="brick-arrow up">
          <div class="arrow-row"><span class="empty"></span><span class="empty"></span><span class="brick"></span><span class="empty"></span><span class="empty"></span></div>
          <div class="arrow-row"><span class="empty"></span><span class="brick"></span><span class="brick"></span><span class="brick"></span><span class="empty"></span></div>
          <div class="arrow-row"><span class="brick"></span><span class="brick"></span><span class="brick"></span><span class="brick"></span><span class="brick"></span></div>
        </div>
      </button>
      <button class="nav-btn arrow-btn" :disabled="currentLayer === 0" @click="prevLayer">
        <div class="brick-arrow down">
          <div class="arrow-row"><span class="brick"></span><span class="brick"></span><span class="brick"></span><span class="brick"></span><span class="brick"></span></div>
          <div class="arrow-row"><span class="empty"></span><span class="brick"></span><span class="brick"></span><span class="brick"></span><span class="empty"></span></div>
          <div class="arrow-row"><span class="empty"></span><span class="empty"></span><span class="brick"></span><span class="empty"></span><span class="empty"></span></div>
        </div>
      </button>
    </div>

    <div class="canvas-container" ref="canvasContainer">
      <canvas ref="canvas"></canvas>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { store } from '../store'

const emit = defineEmits(['update:currentLayer', 'rotateLeft', 'rotateRight'])

const props = defineProps({
  currentLayer: {
    type: Number,
    default: 0
  },
  viewAngle: {
    type: Number,
    default: 0
  },
  isCornerView: {
    type: Boolean,
    default: false
  }
})

const canvas = ref(null)
const canvasContainer = ref(null)
let ctx = null
let resizeObserver = null

const BRICK_COLORS = [
  '#DA291C', // red
  '#0055BF', // blue
  '#237841', // green
  '#FFA500', // orange
  '#FFD700', // yellow
  '#800080', // purple
  '#00CED1'  // cyan
]

const BRICK_DIMS = {
  '1x1': [1, 1], '1x2': [1, 2], '1x3': [1, 3],
  '1x4': [1, 4], '1x6': [1, 6], '2x2': [2, 2],
  '2x3': [2, 3], '2x4': [2, 4], '2x6': [2, 6]
}

const totalLayers = computed(() => store.buildResult?.layers?.length || 0)

const gridSize = computed(() => {
  const dims = store.buildResult?.metadata?.voxel_dimensions
  return dims ? { width: dims[0], depth: dims[1] } : { width: 10, depth: 10 }
})

function prevLayer() {
  if (props.currentLayer > 0) {
    emit('update:currentLayer', props.currentLayer - 1)
  }
}

function nextLayer() {
  if (props.currentLayer < totalLayers.value - 1) {
    emit('update:currentLayer', props.currentLayer + 1)
  }
}

function drawLayer() {
  if (!canvas.value || !ctx || !canvasContainer.value) return

  const canvasEl = canvas.value
  const containerWidth = canvasContainer.value.clientWidth
  const containerHeight = canvasContainer.value.clientHeight

  canvasEl.width = containerWidth
  canvasEl.height = containerHeight

  const width = canvasEl.width
  const height = canvasEl.height
  const grid = gridSize.value

  // Calculate cell size to fit the grid within the container with padding
  const padding = 20
  const availableWidth = width - padding * 2
  const availableHeight = height - padding * 2

  const cellSize = Math.floor(Math.min(
    availableWidth / grid.width,
    availableHeight / grid.depth
  ))

  const gridWidth = grid.width * cellSize
  const gridHeight = grid.depth * cellSize
  const offsetX = (width - gridWidth) / 2
  const offsetY = (height - gridHeight) / 2

  // Clear with yellow background
  ctx.fillStyle = '#FFD700'
  ctx.fillRect(0, 0, width, height)

  // Draw white grid background
  ctx.fillStyle = '#FFFFFF'
  ctx.fillRect(offsetX, offsetY, gridWidth, gridHeight)

  // Draw grid lines
  ctx.strokeStyle = '#ddd'
  ctx.lineWidth = 1
  for (let x = 0; x <= grid.width; x++) {
    ctx.beginPath()
    ctx.moveTo(offsetX + x * cellSize, offsetY)
    ctx.lineTo(offsetX + x * cellSize, offsetY + gridHeight)
    ctx.stroke()
  }
  for (let y = 0; y <= grid.depth; y++) {
    ctx.beginPath()
    ctx.moveTo(offsetX, offsetY + y * cellSize)
    ctx.lineTo(offsetX + gridWidth, offsetY + y * cellSize)
    ctx.stroke()
  }

  // Draw bricks
  const layer = store.buildResult?.layers?.[props.currentLayer] || []
  layer.forEach((brick, i) => {
    const dims = BRICK_DIMS[brick.type] || [1, 1]
    const brickWidth = dims[0]  // width (smaller dimension)
    const brickLength = dims[1] // length (larger dimension)

    // Backend convention: rotation 0 = length along X, rotation 90 = length along Y
    let w, h
    if (brick.rotation === 90) {
      w = brickWidth   // X = width
      h = brickLength  // Y = length
    } else {
      w = brickLength  // X = length
      h = brickWidth   // Y = width
    }

    const bx = offsetX + brick.x * cellSize
    const by = offsetY + brick.y * cellSize
    const bw = w * cellSize
    const bh = h * cellSize

    // Brick fill
    ctx.fillStyle = BRICK_COLORS[i % BRICK_COLORS.length]
    ctx.fillRect(bx + 2, by + 2, bw - 4, bh - 4)

    // Brick border
    ctx.strokeStyle = '#000'
    ctx.lineWidth = 2
    ctx.strokeRect(bx + 2, by + 2, bw - 4, bh - 4)

    // Draw studs
    ctx.fillStyle = darkenColor(BRICK_COLORS[i % BRICK_COLORS.length], 0.2)
    const studRadius = Math.max(4, cellSize * 0.2)
    for (let sx = 0; sx < w; sx++) {
      for (let sy = 0; sy < h; sy++) {
        const cx = bx + (sx + 0.5) * cellSize
        const cy = by + (sy + 0.5) * cellSize
        ctx.beginPath()
        ctx.arc(cx, cy, studRadius, 0, Math.PI * 2)
        ctx.fill()
      }
    }
  })

  // Draw grid border
  ctx.strokeStyle = '#000'
  ctx.lineWidth = 2
  ctx.strokeRect(offsetX, offsetY, gridWidth, gridHeight)

  if (props.isCornerView) {
    // Draw bold black dot at the corner we're viewing from
    ctx.fillStyle = '#000'
    let dotX, dotY
    const dotRadius = 8

    switch(props.viewAngle) {
      case 0: // Front view - viewing from +Y (bottom right corner)
        dotX = offsetX + gridWidth
        dotY = offsetY + gridHeight
        break
      case 1: // Right view - viewing from +X (top right corner)
        dotX = offsetX + gridWidth
        dotY = offsetY
        break
      case 2: // Back view - viewing from -Y (top left corner)
        dotX = offsetX
        dotY = offsetY
        break
      case 3: // Left view - viewing from -X (bottom left corner)
        dotX = offsetX
        dotY = offsetY + gridHeight
        break
    }

    ctx.beginPath()
    ctx.arc(dotX, dotY, dotRadius, 0, Math.PI * 2)
    ctx.fill()

    // Add white outline for visibility
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 2
    ctx.stroke()
  } else {
    // Draw bold black line on the edge being viewed from
    ctx.strokeStyle = '#000'
    ctx.lineWidth = 6
    ctx.beginPath()

    switch(props.viewAngle) {
      case 0: // Front view - top edge (viewing from +Y)
        ctx.moveTo(offsetX, offsetY)
        ctx.lineTo(offsetX + gridWidth, offsetY)
        break
      case 1: // Right view - left edge (viewing from +X)
        ctx.moveTo(offsetX, offsetY)
        ctx.lineTo(offsetX, offsetY + gridHeight)
        break
      case 2: // Back view - bottom edge (viewing from -Y)
        ctx.moveTo(offsetX, offsetY + gridHeight)
        ctx.lineTo(offsetX + gridWidth, offsetY + gridHeight)
        break
      case 3: // Left view - right edge (viewing from -X)
        ctx.moveTo(offsetX + gridWidth, offsetY)
        ctx.lineTo(offsetX + gridWidth, offsetY + gridHeight)
        break
    }

    ctx.stroke()
  }
}

function darkenColor(hex, amount) {
  const num = parseInt(hex.slice(1), 16)
  const r = Math.max(0, (num >> 16) - Math.floor(255 * amount))
  const g = Math.max(0, ((num >> 8) & 0x00FF) - Math.floor(255 * amount))
  const b = Math.max(0, (num & 0x0000FF) - Math.floor(255 * amount))
  return `rgb(${r},${g},${b})`
}

function handleKeydown(e) {
  if (e.key === 'ArrowDown') prevLayer()
  if (e.key === 'ArrowUp') nextLayer()
  if (e.key === 'ArrowLeft') {
    e.preventDefault()
    // Emit rotation event to parent
    emit('rotateLeft')
  }
  if (e.key === 'ArrowRight') {
    e.preventDefault()
    // Emit rotation event to parent
    emit('rotateRight')
  }
}

watch(() => props.currentLayer, drawLayer)
watch(() => props.viewAngle, drawLayer)
watch(() => store.buildResult, drawLayer)

onMounted(() => {
  ctx = canvas.value.getContext('2d')

  // Use ResizeObserver for dynamic sizing
  resizeObserver = new ResizeObserver(() => {
    drawLayer()
  })
  resizeObserver.observe(canvasContainer.value)

  drawLayer()
  window.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeydown)
  if (resizeObserver) {
    resizeObserver.disconnect()
  }
})
</script>

<style scoped>
.layer-viewer {
  height: 100%;
  display: flex;
  flex-direction: column;
  background: var(--lego-white);
  border: 3px solid var(--lego-black);
  border-radius: var(--radius-lg);
  overflow: hidden;
}

.header {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--spacing-lg);
  padding: var(--spacing-md);
  background: #f5f5f5;
  border-bottom: 3px solid var(--lego-black);
}

.nav-btn {
  background: var(--lego-white);
  border: 3px solid var(--lego-black);
  border-radius: var(--radius-md);
  padding: 8px;
}

.nav-btn.arrow-btn {
  width: auto;
  height: auto;
}

.nav-btn:hover:not(:disabled) {
  background: #f0f0f0;
}

.nav-btn:disabled {
  opacity: 0.4;
}

.brick-arrow {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.arrow-row {
  display: flex;
  gap: 2px;
}

.brick-arrow .brick {
  width: 12px;
  height: 12px;
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
  width: 6px;
  height: 6px;
  background: #b8231a;
  border-radius: 50%;
}



.brick-arrow .empty {
  width: 12px;
  height: 12px;
}

.nav-btn:disabled .brick-arrow .brick {
  background: #ccc;
}

.nav-btn:disabled .brick-arrow .brick::after {
  background: #aaa;
}



.layer-indicator {
  font-size: 18px;
  font-weight: 700;
  min-width: 150px;
  text-align: center;
}

.canvas-container {
  flex: 1;
  min-height: 0;
  position: relative;
}

canvas {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: block;
}
</style>

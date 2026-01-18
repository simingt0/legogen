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

const emit = defineEmits(['update:currentLayer', 'viewAngleChange'])

const canvas = ref(null)
const viewContainer = ref(null)
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

  // Reset hitboxes
  layerHitboxes = []

  // Draw each layer from bottom to top
  for (let layerIdx = 0; layerIdx < totalLayers.value; layerIdx++) {
    const layer = layers.value[layerIdx]
    const isCurrent = layerIdx === props.currentLayer

    // Calculate Y position (bottom layer at bottom)
    const y = startY + (totalLayers.value - 1 - layerIdx) * cellSize

    // Store hitbox for click detection
    layerHitboxes.push({
      layerIdx,
      y,
      height: cellSize,
      x: startX,
      width: totalWidth
    })

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
  if (!canvas.value) return

  const rect = canvas.value.getBoundingClientRect()
  const x = event.clientX - rect.left
  const y = event.clientY - rect.top

  // Check if click hit any layer
  for (const hitbox of layerHitboxes) {
    if (x >= hitbox.x && x <= hitbox.x + hitbox.width &&
        y >= hitbox.y && y <= hitbox.y + hitbox.height) {
      emit('update:currentLayer', hitbox.layerIdx)
      break
    }
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

watch(() => props.currentLayer, drawSideView)
watch(() => props.viewAngle, drawSideView)
watch(() => store.buildResult, drawSideView, { deep: true })

onMounted(() => {
  ctx = canvas.value.getContext('2d')

  resizeObserver = new ResizeObserver(() => {
    drawSideView()
  })
  resizeObserver.observe(viewContainer.value)

  drawSideView()
})

onUnmounted(() => {
  if (resizeObserver) {
    resizeObserver.disconnect()
  }
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
</style>

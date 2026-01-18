<template>
  <div class="inventory-sidebar">
    <h2 class="title">Inventory</h2>

    <div v-if="currentLayerPieces.length > 0" class="section">
      <h3 class="section-title">Layer {{ currentLayer + 1 }}</h3>
      <ul class="piece-list current">
        <li v-for="(piece, i) in currentLayerPieces" :key="'current-' + i" class="piece-item">
          <div class="brick-icon" :style="getBrickStyle(piece.type)">
            <div
              v-for="stud in getStudPositions(piece.type)"
              :key="`stud-${stud.x}-${stud.y}`"
              class="stud"
              :style="{ left: stud.x + '%', top: stud.y + '%' }"
            ></div>
          </div>
          <span class="piece-name">{{ piece.type }}</span>
          <span class="piece-count">x{{ piece.count }}</span>
        </li>
      </ul>
    </div>

    <div class="divider"></div>

    <div class="section">
      <h3 class="section-title muted">Full Build</h3>
      <ul class="piece-list full">
        <li v-for="(piece, i) in totalPieces" :key="'total-' + i" class="piece-item muted">
          <div class="brick-icon small" :style="getBrickStyle(piece.type)">
            <div
              v-for="stud in getStudPositions(piece.type)"
              :key="`stud-${stud.x}-${stud.y}`"
              class="stud"
              :style="{ left: stud.x + '%', top: stud.y + '%' }"
            ></div>
          </div>
          <span class="piece-name">{{ piece.type }}</span>
          <span class="piece-count">x{{ piece.count }}</span>
        </li>
      </ul>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { store } from '../store'

const props = defineProps({
  currentLayer: {
    type: Number,
    default: 0
  }
})

const BRICK_COLORS = {
  '1x1': '#DA291C',
  '1x2': '#0055BF',
  '1x3': '#237841',
  '1x4': '#FFA500',
  '1x6': '#FFD700',
  '2x2': '#DA291C',
  '2x3': '#0055BF',
  '2x4': '#237841',
  '2x6': '#FFA500'
}

const BRICK_DIMS = {
  '1x1': [1, 1], '1x2': [1, 2], '1x3': [1, 3],
  '1x4': [1, 4], '1x6': [1, 6], '2x2': [2, 2],
  '2x3': [2, 3], '2x4': [2, 4], '2x6': [2, 6]
}

function getBrickColor(type) {
  return BRICK_COLORS[type] || '#888'
}

function getBrickStyle(type) {
  const dims = BRICK_DIMS[type] || [1, 1]
  const [width, length] = dims

  // Base unit size in pixels (for width dimension)
  const unitSize = 12

  return {
    width: `${length * unitSize}px`,
    height: `${width * unitSize}px`,
    backgroundColor: '#666'  // Dark gray for all inventory bricks
  }
}

function getStudPositions(type) {
  const dims = BRICK_DIMS[type] || [1, 1]
  const [width, length] = dims

  const studs = []
  for (let w = 0; w < width; w++) {
    for (let l = 0; l < length; l++) {
      studs.push({
        x: ((l + 0.5) / length) * 100,
        y: ((w + 0.5) / width) * 100
      })
    }
  }
  return studs
}

const currentLayerPieces = computed(() => {
  if (!store.buildResult?.layers?.[props.currentLayer]) return []

  const counts = {}
  for (const brick of store.buildResult.layers[props.currentLayer]) {
    counts[brick.type] = (counts[brick.type] || 0) + 1
  }

  return Object.entries(counts)
    .map(([type, count]) => ({ type, count }))
    .sort((a, b) => b.count - a.count)
})

const totalPieces = computed(() => {
  if (!store.buildResult?.layers) return []

  const counts = {}
  for (const layer of store.buildResult.layers) {
    for (const brick of layer) {
      counts[brick.type] = (counts[brick.type] || 0) + 1
    }
  }

  return Object.entries(counts)
    .map(([type, count]) => ({ type, count }))
    .sort((a, b) => b.count - a.count)
})
</script>

<style scoped>
.inventory-sidebar {
  height: 100%;
  background: var(--lego-white);
  border: 3px solid var(--lego-black);
  border-radius: var(--radius-lg);
  padding: var(--spacing-md);
  display: flex;
  flex-direction: column;
  overflow-y: auto;
}

.title {
  font-size: 18px;
  font-weight: 900;
  color: var(--lego-red);
  margin-bottom: var(--spacing-md);
  text-transform: uppercase;
}

.section {
  margin-bottom: var(--spacing-md);
}

.section-title {
  font-size: 14px;
  font-weight: 700;
  color: var(--lego-black);
  margin-bottom: var(--spacing-sm);
}

.section-title.muted {
  color: #888;
}

.piece-list {
  list-style: none;
}

.piece-item {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm);
  padding: var(--spacing-xs) 0;
}

.piece-item.muted {
  opacity: 0.6;
}

.brick-icon {
  position: relative;
  border-radius: 2px;
  border: 2px solid rgba(0, 0, 0, 0.3);
  flex-shrink: 0;
  box-shadow: inset 0 -1px 2px rgba(0, 0, 0, 0.2);
}

.brick-icon.small {
  transform: scale(0.8);
  transform-origin: left center;
}

.stud {
  position: absolute;
  width: 6px;
  height: 6px;
  background: inherit;
  filter: brightness(0.75);
  border-radius: 50%;
  transform: translate(-50%, -50%);
  box-shadow: inset 0 -1px 1px rgba(0, 0, 0, 0.4);
}

.piece-name {
  flex: 1;
  font-weight: 600;
  font-size: 14px;
}

.piece-count {
  font-weight: 700;
  font-size: 14px;
  color: var(--lego-red);
}

.piece-list.full .piece-count {
  color: #888;
}

.divider {
  height: 2px;
  background: #ddd;
  margin: var(--spacing-md) 0;
}
</style>

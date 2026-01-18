<template>
  <div class="inventory-sidebar">
    <h2 class="title">Inventory</h2>

    <div v-if="currentLayerPieces.length > 0" class="section">
      <h3 class="section-title">Layer {{ currentLayer + 1 }}</h3>
      <ul class="piece-list current">
        <li v-for="(piece, i) in currentLayerPieces" :key="'current-' + i" class="piece-item">
          <span class="piece-icon" :style="{ backgroundColor: getBrickColor(piece.type) }"></span>
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
          <span class="piece-icon small" :style="{ backgroundColor: getBrickColor(piece.type) }"></span>
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

function getBrickColor(type) {
  return BRICK_COLORS[type] || '#888'
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

.piece-icon {
  width: 24px;
  height: 16px;
  border-radius: 2px;
  border: 2px solid rgba(0, 0, 0, 0.3);
}

.piece-icon.small {
  width: 18px;
  height: 12px;
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

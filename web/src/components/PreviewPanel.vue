<template>
  <div class="preview-panel">
    <h2 class="title">Preview</h2>

    <div class="side-view-area">
      <SideView v-model:currentLayer="modelCurrentLayer" :viewAngle="viewAngle" @viewAngleChange="handleViewAngleChange" @update:isCornerView="handleIsCornerViewChange" />
    </div>

    <div class="stats" v-if="metadata">
      <div class="stat">
        <span class="stat-label">Dimensions</span>
        <span class="stat-value">{{ metadata.voxel_dimensions?.join(' x ') }}</span>
      </div>
      <div class="stat">
        <span class="stat-label">Total Bricks</span>
        <span class="stat-value">{{ metadata.total_bricks }}</span>
      </div>
      <div class="stat">
        <span class="stat-label">Layers</span>
        <span class="stat-value">{{ layers }}</span>
      </div>
    </div>

    <div class="controls">
      <button class="control-btn" @click="goToLanding">New Build</button>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { store, resetForNewBuild } from '../store'
import SideView from './SideView.vue'

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

const router = useRouter()

const metadata = computed(() => store.buildResult?.metadata)
const layers = computed(() => store.buildResult?.layers?.length || 0)

const modelCurrentLayer = computed({
  get: () => props.currentLayer,
  set: (value) => emit('update:currentLayer', value)
})

function handleViewAngleChange(angle) {
  emit('viewAngleChange', angle)
}

function handleIsCornerViewChange(isCorner) {
  emit('update:isCornerView', isCorner)
}

function goToLanding() {
  resetForNewBuild()
  router.push('/')
}
</script>

<style scoped>
.preview-panel {
  height: 100%;
  background: var(--lego-white);
  border: 3px solid var(--lego-black);
  border-radius: var(--radius-lg);
  padding: var(--spacing-md);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
}

.title {
  font-size: 18px;
  font-weight: 900;
  color: var(--lego-red);
  text-transform: uppercase;
}

.side-view-area {
  flex: 1;
  min-height: 0;
  display: flex;
}

.control-btn {
  flex: 1;
  padding: var(--spacing-sm) var(--spacing-md);
  font-size: 14px;
  font-weight: 700;
  background: var(--lego-red);
  color: white;
  border: 3px solid var(--lego-black);
}

.control-btn:hover {
  background: #b8231a;
}

.controls {
  display: flex;
  gap: var(--spacing-sm);
}

.stats {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
  padding: var(--spacing-sm) 0;
  border-top: 2px solid #eee;
  border-bottom: 2px solid #eee;
}

.stat {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
}

.stat-label {
  color: #666;
}

.stat-value {
  font-weight: 700;
}
</style>

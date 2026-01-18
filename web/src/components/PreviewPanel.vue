<template>
  <div class="preview-panel">
    <h2 class="title">Preview</h2>

    <div class="preview-area">
      <div class="placeholder">
        <span class="placeholder-text">3D Preview</span>
        <span class="placeholder-subtext">Coming Soon</span>
      </div>
    </div>

    <div class="controls">
      <button class="control-btn" @click="goToLanding">New Build</button>
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
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { store, resetForNewBuild } from '../store'

const router = useRouter()

const metadata = computed(() => store.buildResult?.metadata)
const layers = computed(() => store.buildResult?.layers?.length || 0)

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

.preview-area {
  flex: 1;
  min-height: 200px;
  background: #f0f0f0;
  border: 2px dashed #ccc;
  border-radius: var(--radius-md);
  display: flex;
  align-items: center;
  justify-content: center;
}

.placeholder {
  text-align: center;
}

.placeholder-text {
  display: block;
  font-size: 18px;
  font-weight: 700;
  color: #888;
}

.placeholder-subtext {
  display: block;
  font-size: 14px;
  color: #aaa;
  margin-top: var(--spacing-xs);
}

.controls {
  display: flex;
  gap: var(--spacing-sm);
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

.stats {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-xs);
  padding-top: var(--spacing-sm);
  border-top: 2px solid #eee;
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

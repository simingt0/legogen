<template>
  <div class="building-page">
    <aside class="inventory-panel">
      <InventorySidebar :currentLayer="currentLayer" />
    </aside>

    <main class="layer-panel">
      <LayerViewer v-model:currentLayer="currentLayer" :viewAngle="viewAngle" :isCornerView="isCornerView" @rotateLeft="handleRotateLeft" @rotateRight="handleRotateRight" />
    </main>

    <aside class="preview-panel">
      <PreviewPanel v-model:currentLayer="currentLayer" :viewAngle="viewAngle" @viewAngleChange="handleViewAngleChange" @update:isCornerView="handleIsCornerViewChange" />
    </aside>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import InventorySidebar from '../components/InventorySidebar.vue'
import LayerViewer from '../components/LayerViewer.vue'
import PreviewPanel from '../components/PreviewPanel.vue'
import { store } from '../store'

const router = useRouter()
const currentLayer = ref(0)
const viewAngle = ref(0)
const isCornerView = ref(false)

function handleViewAngleChange(angle) {
  viewAngle.value = angle
}

function handleRotateLeft() {
  viewAngle.value = (viewAngle.value + 3) % 4
}

function handleRotateRight() {
  viewAngle.value = (viewAngle.value + 1) % 4
}

function handleIsCornerViewChange(isCorner) {
  isCornerView.value = isCorner
}

onMounted(() => {
  // Redirect to landing if no build result
  if (!store.buildResult) {
    router.replace('/')
  }
})
</script>

<style scoped>
.building-page {
  height: 100%;
  display: flex;
  gap: var(--spacing-md);
  padding: var(--spacing-md);
}

.inventory-panel {
  width: 220px;
  flex-shrink: 0;
}

.layer-panel {
  flex: 1;
  min-width: 0;
}

.preview-panel {
  width: 360px;
  flex-shrink: 0;
}
</style>

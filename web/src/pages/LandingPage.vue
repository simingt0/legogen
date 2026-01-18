<template>
  <div class="landing-page">
    <LegoTitle />

    <main class="main-content">
      <div class="left-panel">
        <ImageUploader />
      </div>

      <div class="right-panel">
        <DescriptionInput />
      </div>
    </main>

    <footer class="footer">
      <GenerateButton ref="generateBtn" />
    </footer>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import LegoTitle from '../components/LegoTitle.vue'
import ImageUploader from '../components/ImageUploader.vue'
import DescriptionInput from '../components/DescriptionInput.vue'
import GenerateButton from '../components/GenerateButton.vue'
import { store, startBuild, resetForNewBuild } from '../store'

const router = useRouter()
const generateBtn = ref(null)

function handleKeydown(e) {
  // cmd+enter (Mac) or ctrl+enter (Windows/Linux)
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    e.preventDefault()
    triggerGenerate()
  }
}

function triggerGenerate() {
  if (store.imageFile && store.description.trim().length > 0) {
    resetForNewBuild()
    startBuild()
    router.push('/loading')
  }
}

onMounted(() => {
  window.addEventListener('keydown', handleKeydown)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handleKeydown)
})
</script>

<style scoped>
.landing-page {
  height: 100%;
  display: flex;
  flex-direction: column;
  padding: var(--spacing-lg);
  gap: var(--spacing-lg);
}

.main-content {
  flex: 1;
  display: flex;
  gap: var(--spacing-xl);
  min-height: 0;
}

.left-panel {
  flex: 1;
  display: flex;
}

.right-panel {
  flex: 1;
  display: flex;
}

.footer {
  display: flex;
  justify-content: center;
  padding: var(--spacing-md) 0;
}
</style>

<template>
  <button
    class="generate-btn"
    :disabled="!canGenerate"
    @click="handleGenerate"
  >
    GENERATE
  </button>
</template>

<script setup>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { store, startBuild, resetForNewBuild } from '../store'

const router = useRouter()

const canGenerate = computed(() => {
  return store.imageFile && store.description.trim().length > 0
})

async function handleGenerate() {
  if (!canGenerate.value) return

  resetForNewBuild()
  startBuild()
  router.push('/loading')
}
</script>

<style scoped>
.generate-btn {
  font-size: 24px;
  font-weight: 900;
  padding: var(--spacing-md) var(--spacing-xl);
  background: var(--lego-red);
  color: white;
  border: 4px solid var(--lego-black);
  border-radius: var(--radius-lg);
  transition: all 0.15s ease;
  letter-spacing: 2px;
}

.generate-btn:hover:not(:disabled) {
  background: #b8231a;
}

.generate-btn:active:not(:disabled) {
  transform: scale(0.98);
}

.generate-btn:disabled {
  background: #ccc;
  color: #888;
}
</style>

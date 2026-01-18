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
  startBuild() // Fire and forget - the loading page will await this
  router.push('/loading')
}
</script>

<style scoped>
.generate-btn {
  font-size: 24px;
  font-weight: 900;
  padding: var(--spacing-md) var(--spacing-xl);
  background: linear-gradient(180deg, #ff4444 0%, var(--lego-red) 100%);
  color: white;
  border: 4px solid var(--lego-black);
  border-radius: var(--radius-lg);
  box-shadow:
    0 6px 0 #8b0000,
    0 10px 20px rgba(0, 0, 0, 0.3);
  transition: all 0.15s ease;
  letter-spacing: 2px;
}

.generate-btn:hover:not(:disabled) {
  transform: translateY(-4px);
  box-shadow:
    0 10px 0 #8b0000,
    0 16px 30px rgba(0, 0, 0, 0.4);
}

.generate-btn:active:not(:disabled) {
  transform: translateY(2px);
  box-shadow:
    0 2px 0 #8b0000,
    0 4px 10px rgba(0, 0, 0, 0.3);
}

.generate-btn:disabled {
  background: linear-gradient(180deg, #aaa 0%, #888 100%);
  box-shadow:
    0 6px 0 #555,
    0 10px 20px rgba(0, 0, 0, 0.2);
}
</style>

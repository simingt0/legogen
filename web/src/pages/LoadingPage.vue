<template>
  <div class="loading-page">
    <MiniGame :standalone="isStandalone" />
  </div>
</template>

<script setup>
import { onMounted, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import MiniGame from '../components/MiniGame.vue'
import { store } from '../store'

const router = useRouter()
const route = useRoute()

const isStandalone = computed(() => route.path === '/game')

onMounted(() => {
  // Allow standalone game mode
  if (isStandalone.value) {
    return
  }

  // Redirect to landing if no build in progress
  if (!store.buildPromise && !store.isBuilding) {
    router.replace('/')
  }
})
</script>

<style scoped>
.loading-page {
  height: 100%;
  width: 100%;
  overflow: hidden;
}
</style>

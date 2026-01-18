<template>
  <div class="loading-page">
    <MiniGame />
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import { useRouter } from 'vue-router'
import MiniGame from '../components/MiniGame.vue'
import { store } from '../store'

const router = useRouter()

onMounted(() => {
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

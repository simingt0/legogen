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
      <GenerateButton />
    </footer>
  </div>
</template>

<script setup>
import LegoTitle from '../components/LegoTitle.vue'
import ImageUploader from '../components/ImageUploader.vue'
import DescriptionInput from '../components/DescriptionInput.vue'
import GenerateButton from '../components/GenerateButton.vue'
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

<template>
  <div
    class="uploader"
    :class="{ 'has-image': hasImage, 'drag-over': isDragging }"
    @dragover.prevent="isDragging = true"
    @dragleave.prevent="isDragging = false"
    @drop.prevent="handleDrop"
    @click="triggerFileInput"
  >
    <input
      ref="fileInput"
      type="file"
      accept="image/jpeg,image/png"
      @change="handleFileSelect"
      hidden
    />

    <template v-if="hasImage">
      <img :src="store.imagePreviewUrl" alt="Uploaded LEGO bricks" class="preview" />
      <button class="change-btn" @click.stop="clearImage">Change Image</button>
    </template>

    <template v-else>
      <div class="upload-prompt">
        <div class="icon">+</div>
        <p class="title">Upload Your LEGO Bricks</p>
        <p class="subtitle">Drag & drop or click to select</p>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import { store, setImage } from '../store'

const fileInput = ref(null)
const isDragging = ref(false)

const hasImage = computed(() => !!store.imageFile)

function triggerFileInput() {
  if (!hasImage.value) {
    fileInput.value?.click()
  }
}

function handleFileSelect(e) {
  const file = e.target.files?.[0]
  if (file && isValidImage(file)) {
    setImage(file)
  }
}

function handleDrop(e) {
  isDragging.value = false
  const file = e.dataTransfer?.files?.[0]
  if (file && isValidImage(file)) {
    setImage(file)
  }
}

function isValidImage(file) {
  return file.type === 'image/jpeg' || file.type === 'image/png'
}

function clearImage() {
  setImage(null)
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}
</script>

<style scoped>
.uploader {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border: 4px dashed var(--lego-black);
  border-radius: var(--radius-lg);
  background-color: var(--lego-white);
  cursor: pointer;
  transition: all 0.2s ease;
  position: relative;
  overflow: hidden;
}

.uploader:hover:not(.has-image) {
  border-color: var(--lego-red);
  background-color: #fff8f8;
}

.uploader.drag-over {
  border-color: var(--lego-red);
  background-color: #ffeeee;
  transform: scale(1.02);
}

.uploader.has-image {
  border-style: solid;
  cursor: default;
}

.upload-prompt {
  text-align: center;
  padding: var(--spacing-xl);
}

.icon {
  font-size: 64px;
  font-weight: bold;
  color: var(--lego-red);
  line-height: 1;
  margin-bottom: var(--spacing-md);
}

.title {
  font-size: 20px;
  font-weight: 700;
  margin-bottom: var(--spacing-sm);
}

.subtitle {
  font-size: 14px;
  color: #666;
}

.preview {
  width: 100%;
  height: 100%;
  object-fit: contain;
  padding: var(--spacing-md);
}

.change-btn {
  position: absolute;
  bottom: var(--spacing-md);
  right: var(--spacing-md);
  background: var(--lego-red);
  color: white;
  border-color: var(--lego-black);
  font-size: 14px;
}

.change-btn:hover {
  background: #b8231a;
}
</style>

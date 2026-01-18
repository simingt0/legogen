<template>
  <div class="description-input" :class="{ empty: !localDescription.trim() && !isFocused }">
    <div class="input-container">
      <div v-if="!localDescription.trim() && !isFocused" class="placeholder-overlay">
        Describe what you want to build...
      </div>
      <textarea
        v-model="localDescription"
        placeholder=""
        maxlength="120"
        @input="updateDescription"
        @focus="isFocused = true"
        @blur="isFocused = false"
      ></textarea>
      <span class="char-count">{{ localDescription.length }}/120</span>
    </div>

    <div class="suggestions">
      <button
        v-for="suggestion in suggestions"
        :key="suggestion.id"
        class="suggestion-btn"
        :class="{ active: localDescription === suggestion.text }"
        @click="selectSuggestion(suggestion.text)"
      >
        {{ suggestion.label }}
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import { store, setDescription } from '../store'

const suggestions = [
  { id: 1, label: 'Mushroom', text: 'mushroom' },
  { id: 2, label: 'House', text: 'small house' },
  { id: 3, label: 'Mug', text: 'mug' },
  { id: 4, label: 'Banana', text: 'banana' },
  { id: 5, label: 'Dragon', text: 'small dragon' },
  { id: 6, label: 'Creeper', text: 'minecraft creeper' },
  { id: 7, label: 'Castle Tower', text: 'castle tower' },
  { id: 8, label: 'Dwayne the Block Johnson', text: "realistic rendering of dwayne the rock johnson's face" }
]

const localDescription = ref(store.description)
const isFocused = ref(false)

watch(() => store.description, (newVal) => {
  localDescription.value = newVal
})

function updateDescription() {
  setDescription(localDescription.value)
}

function selectSuggestion(text) {
  localDescription.value = text
  setDescription(text)
}
</script>

<style scoped>
.description-input {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md);
  border: 4px dashed var(--lego-black);
  border-radius: var(--radius-lg);
  background-color: var(--lego-white);
  padding: var(--spacing-lg);
  transition: all 0.2s ease;
}

.description-input.empty {
  animation: gentle-pulse 3s ease-in-out infinite;
}

@keyframes gentle-pulse {
  0%, 100% {
    border-color: var(--lego-black);
    opacity: 1;
  }
  50% {
    border-color: var(--lego-red);
    opacity: 0.85;
  }
}

.input-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  position: relative;
}

.placeholder-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  font-weight: 700;
  color: var(--lego-black);
  pointer-events: none;
  text-align: center;
  padding: var(--spacing-md);
}

textarea {
  flex: 1;
  resize: none;
  font-size: 40px;
  line-height: 1.5;
  min-height: 150px;
  border: none;
  background: transparent;
  padding: 0;
}

textarea:focus {
  outline: none;
  border: none;
}

.char-count {
  position: absolute;
  bottom: var(--spacing-sm);
  right: var(--spacing-sm);
  font-size: 12px;
  color: #999;
}

.suggestions {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  grid-template-rows: repeat(2, 1fr);
  gap: var(--spacing-sm);
}

.suggestion-btn {
  background: transparent;
  border: 2px dashed var(--lego-black);
  color: var(--lego-black);
  font-size: 16px;
  padding: var(--spacing-md) var(--spacing-lg);
  transition: all 0.2s ease;
  text-align: center;
  font-weight: 600;
}

.suggestion-btn:hover {
  border-color: var(--lego-red);
  color: var(--lego-red);
  border-style: solid;
}

.suggestion-btn.active {
  background: var(--lego-red);
  border-color: var(--lego-red);
  border-style: solid;
  color: white;
}
</style>

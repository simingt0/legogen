<template>
  <div class="description-input">
    <div class="input-container">
      <textarea
        v-model="localDescription"
        placeholder="Describe what you want to build..."
        maxlength="600"
        @input="updateDescription"
      ></textarea>
      <span class="char-count">{{ localDescription.length }}/600</span>
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
  { id: 1, label: 'Mushroom', text: 'A cute red mushroom with white spots' },
  { id: 2, label: 'House', text: 'A simple house with a door and windows' },
  { id: 3, label: 'Mug', text: 'A coffee mug with a handle' },
  { id: 4, label: 'Banana', text: 'A yellow banana' },
  { id: 5, label: 'Dwayne the Block Johnson', text: 'Dwayne "The Rock" Johnson as a blocky figure' }
]

const localDescription = ref(store.description)

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
}

.input-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  position: relative;
}

textarea {
  flex: 1;
  resize: none;
  font-size: 16px;
  line-height: 1.5;
  min-height: 150px;
}

.char-count {
  position: absolute;
  bottom: var(--spacing-sm);
  right: var(--spacing-sm);
  font-size: 12px;
  color: #999;
}

.suggestions {
  display: flex;
  flex-wrap: wrap;
  gap: var(--spacing-sm);
}

.suggestion-btn {
  background: transparent;
  border: 2px dashed #ccc;
  color: #888;
  font-size: 14px;
  padding: var(--spacing-sm) var(--spacing-md);
  transition: all 0.2s ease;
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

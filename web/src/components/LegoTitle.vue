<template>
  <header class="lego-title">
    <div class="bricks">
      <span v-for="(letter, i) in letters" :key="i" class="brick" :style="{ backgroundColor: brickColors[i % brickColors.length] }">
        {{ letter }}
      </span>
    </div>
  </header>
</template>

<script setup>
const letters = ['L', 'E', 'G', 'O', 'G', 'E', 'N']
const brickColors = ['#DA291C', '#0055BF', '#237841', '#FFD700', '#DA291C', '#0055BF', '#237841']
</script>

<style scoped>
.lego-title {
  text-align: center;
  padding: var(--spacing-md) 0;
}

.bricks {
  display: inline-flex;
  gap: 6px;
}

.brick {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 64px;
  height: 72px;
  font-size: 36px;
  font-weight: 900;
  color: white;
  text-shadow: 2px 2px 0 rgba(0, 0, 0, 0.3);
  border-radius: 6px;
  position: relative;
  box-shadow:
    inset 0 -4px 0 rgba(0, 0, 0, 0.2),
    0 4px 8px rgba(0, 0, 0, 0.3);
}

/* Stud on top of brick */
.brick::before {
  content: '';
  position: absolute;
  top: -8px;
  left: 50%;
  transform: translateX(-50%);
  width: 24px;
  height: 8px;
  background: inherit;
  border-radius: 50% 50% 0 0;
  box-shadow:
    inset 0 -2px 0 rgba(0, 0, 0, 0.2),
    0 -2px 4px rgba(0, 0, 0, 0.1);
}
</style>

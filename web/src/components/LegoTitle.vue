<template>
  <header class="lego-title">
    <div class="brick-grid">
      <div
        v-for="(row, y) in grid"
        :key="y"
        class="brick-row"
      >
        <div
          v-for="(cell, x) in row"
          :key="x"
          class="brick-cell"
          :class="{ filled: cell }"
          :style="cell ? { backgroundColor: getColor(x, y) } : {}"
        >
          <div v-if="cell" class="stud"></div>
        </div>
      </div>
    </div>
  </header>
</template>

<script setup>
// Each letter is defined as a small grid (5 tall x variable width)
// 1 = filled, 0 = empty
const letterPatterns = {
  L: [
    [1,0,0],
    [1,0,0],
    [1,0,0],
    [1,0,0],
    [1,1,1],
  ],
  E: [
    [1,1,1],
    [1,0,0],
    [1,1,0],
    [1,0,0],
    [1,1,1],
  ],
  G: [
    [1,1,1],
    [1,0,0],
    [1,0,1],
    [1,0,1],
    [1,1,1],
  ],
  O: [
    [1,1,1],
    [1,0,1],
    [1,0,1],
    [1,0,1],
    [1,1,1],
  ],
  N: [
    [1,0,1],
    [1,1,1],
    [1,1,1],
    [1,0,1],
    [1,0,1],
  ],
}

const letters = ['L', 'E', 'G', 'O', 'G', 'E', 'N']
const colors = ['#DA291C', '#0055BF', '#237841', '#FFD700', '#DA291C', '#0055BF', '#237841']

// Build the full grid by combining letter patterns with spacing
function buildGrid() {
  const height = 5
  const rows = []

  for (let y = 0; y < height; y++) {
    const row = []
    letters.forEach((letter, letterIdx) => {
      const pattern = letterPatterns[letter]
      for (let x = 0; x < pattern[y].length; x++) {
        row.push(pattern[y][x] ? letterIdx + 1 : 0) // Store letter index + 1 for color
      }
      // Add spacing between letters (except after last)
      if (letterIdx < letters.length - 1) {
        row.push(0)
      }
    })
    rows.push(row)
  }

  return rows
}

const grid = buildGrid()

function getColor(x, y) {
  // Find which letter this cell belongs to by counting columns
  let col = 0
  for (let i = 0; i < letters.length; i++) {
    const pattern = letterPatterns[letters[i]]
    const width = pattern[0].length
    if (x < col + width) {
      return colors[i]
    }
    col += width + 1 // +1 for spacing
  }
  return colors[0]
}
</script>

<style scoped>
.lego-title {
  text-align: center;
  padding: var(--spacing-md) 0;
}

.brick-grid {
  display: inline-flex;
  flex-direction: column;
  gap: 2px;
}

.brick-row {
  display: flex;
  gap: 2px;
}

.brick-cell {
  width: 24px;
  height: 24px;
  border-radius: 2px;
  position: relative;
}

.brick-cell.filled {
  box-shadow: inset 0 -2px 0 rgba(0, 0, 0, 0.2);
}

.stud {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 12px;
  height: 12px;
  background: inherit;
  filter: brightness(0.85);
  border-radius: 50%;
  box-shadow:
    inset 0 -2px 2px rgba(0, 0, 0, 0.3),
    0 1px 2px rgba(0, 0, 0, 0.2);
}
</style>

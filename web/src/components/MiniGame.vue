<template>
  <div class="mini-game" ref="container">
    <div class="score">Coins: {{ coins }}</div>
    <canvas ref="canvas"></canvas>
    <div v-if="showControls" class="controls-hint">
      <div class="key">W</div>
      <div class="key-row">
        <div class="key">A</div>
        <div class="key">S</div>
        <div class="key">D</div>
      </div>
    </div>
    <div v-if="showError" class="error-message">
      Failed to generate a valid structure
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { store } from '../store'

const router = useRouter()
const container = ref(null)
const canvas = ref(null)
const coins = ref(0)
const showControls = ref(true)
const showError = ref(false)

// Game constants
const PILLAR_WIDTH = 80
const PLAYER_SIZE = 40
const COIN_SIZE = 16
const GRAVITY = 0.6
const JUMP_FORCE = -14
const MOVE_SPEED = 6

// Game state
let ctx = null
let animationId = null
let gameState = 'playing'
let transitionProgress = 0

// Player state
const player = {
  x: 0,
  y: 0,
  vy: 0,
  onGround: false
}

// Input state
const keys = {
  left: false,
  right: false,
  up: false,
  down: false
}

// Pillars
let pillars = []

// Coins
let coinList = []

function initGame() {
  const width = container.value.clientWidth
  const height = container.value.clientHeight
  canvas.value.width = width
  canvas.value.height = height
  ctx = canvas.value.getContext('2d')

  // Calculate how many pillars we need to span full width (no gaps)
  const pillarCount = Math.ceil(width / PILLAR_WIDTH) + 2

  pillars = []
  for (let i = 0; i < pillarCount; i++) {
    pillars.push({
      x: i * PILLAR_WIDTH,
      baseY: height * 0.6,
      phase: Math.random() * Math.PI * 2,
      amplitude: 50 + Math.random() * 50,
      speed: 0.015 + Math.random() * 0.025,
      targetSpeed: 0.015 + Math.random() * 0.025,
      y: 0
    })
  }

  // Initialize player on middle pillar
  const middlePillar = pillars[Math.floor(pillarCount / 2)]
  player.x = middlePillar.x + PILLAR_WIDTH / 2 - PLAYER_SIZE / 2
  player.y = middlePillar.baseY - PLAYER_SIZE - 100
  player.vy = 0

  // Initialize coins
  spawnCoins()
}

function spawnCoins() {
  coinList = []
  for (let i = 0; i < pillars.length; i++) {
    if (Math.random() > 0.4) {
      const pillar = pillars[i]
      coinList.push({
        x: pillar.x + PILLAR_WIDTH / 2 - COIN_SIZE / 2,
        baseY: pillar.baseY - 70 - Math.random() * 50,
        pillarIndex: i,
        collected: false
      })
    }
  }
}

function update() {
  const width = canvas.value.width
  const height = canvas.value.height

  if (gameState === 'playing') {
    // Update pillars
    for (const pillar of pillars) {
      // Occasionally change speed
      if (Math.random() < 0.002) {
        pillar.targetSpeed = 0.01 + Math.random() * 0.03
      }
      // Smoothly transition to target speed
      pillar.speed += (pillar.targetSpeed - pillar.speed) * 0.02

      pillar.phase += pillar.speed
      pillar.y = pillar.baseY + Math.sin(pillar.phase) * pillar.amplitude
    }

    // Update coin positions to follow their pillars
    for (const coin of coinList) {
      if (!coin.collected) {
        const pillar = pillars[coin.pillarIndex]
        coin.y = pillar.y - 70 - (coin.baseY - pillar.baseY + 70)
      }
    }

    // Player horizontal movement
    if (keys.left) player.x -= MOVE_SPEED
    if (keys.right) player.x += MOVE_SPEED

    // Keep player in bounds
    player.x = Math.max(0, Math.min(width - PLAYER_SIZE, player.x))

    // Gravity
    player.vy += GRAVITY
    if (keys.down) player.vy += 0.5
    player.y += player.vy

    // Collision with pillars
    player.onGround = false
    for (const pillar of pillars) {
      const pillarTop = pillar.y
      const pillarLeft = pillar.x
      const pillarRight = pillar.x + PILLAR_WIDTH

      if (
        player.vy >= 0 &&
        player.x + PLAYER_SIZE > pillarLeft &&
        player.x < pillarRight &&
        player.y + PLAYER_SIZE >= pillarTop &&
        player.y + PLAYER_SIZE <= pillarTop + player.vy + 10
      ) {
        player.y = pillarTop - PLAYER_SIZE
        player.vy = 0
        player.onGround = true
      }
    }

    // Jump
    if (keys.up && player.onGround) {
      player.vy = JUMP_FORCE
      player.onGround = false
    }

    // Collect coins
    for (const coin of coinList) {
      if (!coin.collected) {
        const dx = (player.x + PLAYER_SIZE / 2) - (coin.x + COIN_SIZE / 2)
        const dy = (player.y + PLAYER_SIZE / 2) - (coin.y + COIN_SIZE / 2)
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < PLAYER_SIZE / 2 + COIN_SIZE / 2) {
          coin.collected = true
          coins.value++
          store.coinsCollected = coins.value
        }
      }
    }

    // Respawn coins if all collected
    if (coinList.every(c => c.collected)) {
      spawnCoins()
    }

    // Check if player fell off
    if (player.y > height + 100) {
      const randomPillar = pillars[Math.floor(Math.random() * pillars.length)]
      player.x = randomPillar.x + PILLAR_WIDTH / 2 - PLAYER_SIZE / 2
      player.y = -PLAYER_SIZE
      player.vy = 0
    }

  } else if (gameState === 'success') {
    transitionProgress += 0.02
    for (const pillar of pillars) {
      pillar.y -= 15
    }
    player.y -= 15

    if (transitionProgress >= 1) {
      router.push('/build')
      return
    }

  } else if (gameState === 'failure') {
    transitionProgress += 0.015
    for (const pillar of pillars) {
      pillar.y += 8
    }
    player.vy += GRAVITY
    player.y += player.vy

    if (transitionProgress >= 1) {
      router.push('/')
      return
    }
  }
}

function draw() {
  const width = canvas.value.width
  const height = canvas.value.height

  // Clear with yellow background
  ctx.fillStyle = '#FFD700'
  ctx.fillRect(0, 0, width, height)

  // Red overlay for success transition
  if (gameState === 'success') {
    ctx.fillStyle = `rgba(218, 41, 28, ${transitionProgress})`
    ctx.fillRect(0, 0, width, height)
  }

  // Draw pillars (continuous, no gaps)
  for (const pillar of pillars) {
    ctx.fillStyle = '#DA291C'
    // Extend slightly to avoid any sub-pixel gaps
    ctx.fillRect(pillar.x - 1, pillar.y, PILLAR_WIDTH + 2, height - pillar.y + 200)

    // Pillar studs (2 studs per pillar width)
    ctx.fillStyle = '#b8231a'
    const studSize = 16
    const studY = pillar.y - 4
    ctx.beginPath()
    ctx.ellipse(pillar.x + PILLAR_WIDTH / 4, studY, studSize / 2, 6, 0, 0, Math.PI * 2)
    ctx.fill()
    ctx.beginPath()
    ctx.ellipse(pillar.x + (PILLAR_WIDTH * 3) / 4, studY, studSize / 2, 6, 0, 0, Math.PI * 2)
    ctx.fill()
  }

  // Draw coins
  for (const coin of coinList) {
    if (!coin.collected) {
      ctx.fillStyle = '#FFD700'
      ctx.strokeStyle = '#000'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.arc(coin.x + COIN_SIZE / 2, coin.y + COIN_SIZE / 2, COIN_SIZE / 2, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()

      ctx.fillStyle = '#FFA500'
      ctx.beginPath()
      ctx.arc(coin.x + COIN_SIZE / 2, coin.y + COIN_SIZE / 2, COIN_SIZE / 4, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  // Draw player (blue 2x2 brick)
  ctx.fillStyle = '#0055BF'
  ctx.fillRect(player.x, player.y, PLAYER_SIZE, PLAYER_SIZE)

  // Player studs
  ctx.fillStyle = '#004494'
  const pStudSize = 12
  const positions = [
    [player.x + PLAYER_SIZE / 4, player.y - 3],
    [player.x + (PLAYER_SIZE * 3) / 4, player.y - 3]
  ]
  for (const [sx, sy] of positions) {
    ctx.beginPath()
    ctx.ellipse(sx, sy, pStudSize / 2, 4, 0, 0, Math.PI * 2)
    ctx.fill()
  }

  // Border on player
  ctx.strokeStyle = '#003366'
  ctx.lineWidth = 2
  ctx.strokeRect(player.x, player.y, PLAYER_SIZE, PLAYER_SIZE)
}

function gameLoop() {
  update()
  draw()
  animationId = requestAnimationFrame(gameLoop)
}

function handleKeyDown(e) {
  if (gameState !== 'playing') return

  const key = e.key.toLowerCase()
  if (key === 'a' || key === 'arrowleft') {
    keys.left = true
    showControls.value = false
  }
  if (key === 'd' || key === 'arrowright') {
    keys.right = true
    showControls.value = false
  }
  if (key === 'w' || key === 'arrowup' || key === ' ') {
    keys.up = true
    showControls.value = false
  }
  if (key === 's' || key === 'arrowdown') {
    keys.down = true
    showControls.value = false
  }
}

function handleKeyUp(e) {
  const key = e.key.toLowerCase()
  if (key === 'a' || key === 'arrowleft') keys.left = false
  if (key === 'd' || key === 'arrowright') keys.right = false
  if (key === 'w' || key === 'arrowup' || key === ' ') keys.up = false
  if (key === 's' || key === 'arrowdown') keys.down = false
}

function handleResize() {
  if (container.value && canvas.value) {
    canvas.value.width = container.value.clientWidth
    canvas.value.height = container.value.clientHeight
    // Reinitialize pillars to span new width
    initGame()
  }
}

onMounted(async () => {
  initGame()
  gameLoop()

  window.addEventListener('keydown', handleKeyDown)
  window.addEventListener('keyup', handleKeyUp)
  window.addEventListener('resize', handleResize)

  try {
    await store.buildPromise
    gameState = 'success'
    transitionProgress = 0
  } catch (error) {
    gameState = 'failure'
    transitionProgress = 0
    showError.value = true
  }
})

onUnmounted(() => {
  if (animationId) {
    cancelAnimationFrame(animationId)
  }
  window.removeEventListener('keydown', handleKeyDown)
  window.removeEventListener('keyup', handleKeyUp)
  window.removeEventListener('resize', handleResize)
})
</script>

<style scoped>
.mini-game {
  width: 100%;
  height: 100%;
  position: relative;
  overflow: hidden;
}

canvas {
  display: block;
  width: 100%;
  height: 100%;
}

.score {
  position: absolute;
  top: var(--spacing-md);
  left: var(--spacing-md);
  font-size: 24px;
  font-weight: 900;
  color: var(--lego-black);
  background: var(--lego-white);
  padding: var(--spacing-sm) var(--spacing-md);
  border: 3px solid var(--lego-black);
  border-radius: var(--radius-md);
  z-index: 10;
}

.controls-hint {
  position: absolute;
  bottom: var(--spacing-xl);
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  z-index: 10;
  animation: pulse 1.5s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 0.7; }
  50% { opacity: 1; }
}

.key-row {
  display: flex;
  gap: 4px;
}

.key {
  width: 48px;
  height: 48px;
  background: var(--lego-white);
  border: 3px solid var(--lego-black);
  border-radius: var(--radius-sm);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  font-weight: 900;
  box-shadow: 0 4px 0 #888;
}

.error-message {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 28px;
  font-weight: 900;
  color: var(--lego-white);
  background: var(--lego-red);
  padding: var(--spacing-lg) var(--spacing-xl);
  border: 4px solid var(--lego-black);
  border-radius: var(--radius-lg);
  z-index: 20;
  text-align: center;
}
</style>

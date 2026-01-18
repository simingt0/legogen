<template>
  <div class="mini-game" ref="container">
    <div class="score">Coins: {{ coins }}</div>
    <div class="falls">Falls: {{ falls }}</div>
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

const props = defineProps({
  standalone: {
    type: Boolean,
    default: false
  }
})

const router = useRouter()
const container = ref(null)
const canvas = ref(null)
const coins = ref(0)
const falls = ref(0)
const showControls = ref(true)
const showError = ref(false)

// Game constants
const PILLAR_WIDTH = 80
const PLAYER_SIZE = 40
const COIN_SIZE = 16
const BIG_COIN_SIZE = 24
const GRAVITY = 0.6
const JUMP_FORCE = -14
const MOVE_SPEED = 6
const STUD_SIZE = 5.5 // Realistic stud radius
const PILLAR_GAP = 2 // Tiny gap between normal pillars
const MAGNET_RADIUS = 80 // Radius for coin magnet effect

// Game state
let ctx = null
let animationId = null
let gameState = 'playing'
let transitionProgress = 0
let pillarDropping = false
let pillarDropProgress = 0

// Player state
const player = {
  x: 0,
  y: 0,
  vy: 0,
  onGround: false,
  hasDoubleJump: false,
  doubleJumpUsed: false,
  hasMagnet: false,
  magnetEndTime: 0,
  hasSpawnedDoubleJump: false // Track if double jump has spawned this game
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

  // Fixed 18 pillars (counting doubles as 2) spanning the screen
  const totalPillarUnits = 18
  const numDoublePillars = 2
  const numNormalPillars = totalPillarUnits - (numDoublePillars * 2)
  const totalPillars = numNormalPillars + numDoublePillars

  // Calculate pillar width to fit screen exactly
  const totalGaps = (totalPillars - 1) * PILLAR_GAP
  const availableWidth = width - totalGaps
  const unitWidth = availableWidth / totalPillarUnits

  pillars = []

  // Choose 2 random indices for double-width pillars
  const doubleWidthIndices = new Set()
  while (doubleWidthIndices.size < 2) {
    const idx = Math.floor(Math.random() * totalPillars)
    if (idx > 0 && idx < totalPillars - 1) { // Not first or last
      doubleWidthIndices.add(idx)
    }
  }

  let currentX = 0
  for (let i = 0; i < totalPillars; i++) {
    const isDoubleWidth = doubleWidthIndices.has(i)
    const pillarWidth = isDoubleWidth ? unitWidth * 2 : unitWidth

    pillars.push({
      x: currentX,
      baseY: height * 0.6,
      phase: Math.random() * Math.PI * 2,
      amplitude: 50 + Math.random() * 50,
      speed: 0.015 + Math.random() * 0.025,
      targetSpeed: 0.015 + Math.random() * 0.025,
      y: 0,
      width: pillarWidth,
      isDoubleWidth: isDoubleWidth
    })

    currentX += pillarWidth + PILLAR_GAP
  }

  // Initialize player on middle pillar
  const middlePillar = pillars[Math.floor(totalPillars / 2)]
  player.x = middlePillar.x + middlePillar.width / 2 - PLAYER_SIZE / 2
  player.y = middlePillar.baseY - PLAYER_SIZE - 100
  player.vy = 0

  // Initialize coins
  spawnCoins()
}

function spawnCoins() {
  coinList = []

  // Decide powerups for this batch
  const spawnDoubleJump = !player.hasSpawnedDoubleJump && Math.random() < 0.15 // 15% if not spawned yet
  const spawnMagnet = Math.random() < 0.15 // 15% chance
  const spawnPillarDrop = Math.random() < 0.15 // 15% chance

  let doubleJumpSpawned = false
  let magnetSpawned = false
  let pillarDropSpawned = false

  for (let i = 0; i < pillars.length; i++) {
    if (Math.random() > 0.4) {
      const pillar = pillars[i]
      let type = 'normal'
      let size = COIN_SIZE
      let value = 1

      // Try to spawn powerups first
      if (spawnDoubleJump && !doubleJumpSpawned) {
        type = 'doubleJump'
        size = COIN_SIZE
        doubleJumpSpawned = true
        player.hasSpawnedDoubleJump = true
      } else if (spawnMagnet && !magnetSpawned) {
        type = 'magnet'
        size = COIN_SIZE
        magnetSpawned = true
      } else if (spawnPillarDrop && !pillarDropSpawned) {
        type = 'pillarDrop'
        size = COIN_SIZE
        pillarDropSpawned = true
      } else if (Math.random() < 0.05) { // 5% big coin
        type = 'big'
        size = BIG_COIN_SIZE
        value = 5
      }

      coinList.push({
        x: pillar.x + pillar.width / 2 - size / 2,
        baseY: pillar.baseY - 70 - Math.random() * 50,
        pillarIndex: i,
        collected: false,
        type: type,
        size: size,
        value: value
      })
    }
  }
}

function update() {
  const width = canvas.value.width
  const height = canvas.value.height

  if (gameState === 'playing') {
    // Update pillars
    if (!pillarDropping) {
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
    } else {
      // Animate pillars dropping
      pillarDropProgress += 0.015
      for (const pillar of pillars) {
        const targetY = height + 200 // Drop below screen
        pillar.y = pillar.y + (targetY - pillar.y) * 0.05
      }

      // When animation complete, reset the game
      if (pillarDropProgress >= 1) {
        pillarDropping = false
        pillarDropProgress = 0
        initGame()
      }
    }

    // Update coin positions to follow their pillars
    for (const coin of coinList) {
      if (!coin.collected) {
        const pillar = pillars[coin.pillarIndex]
        // Move coin with pillar: base position + pillar's offset from rest
        coin.y = coin.baseY + (pillar.y - pillar.baseY)
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
      const pillarRight = pillar.x + pillar.width

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
      player.doubleJumpUsed = false
    } else if (keys.up && player.hasDoubleJump && !player.doubleJumpUsed && !player.onGround && player.vy > 0) {
      // Double jump - only when falling
      player.vy = JUMP_FORCE
      player.doubleJumpUsed = true
    }

    // Check magnet timer
    if (player.hasMagnet && Date.now() > player.magnetEndTime) {
      player.hasMagnet = false
    }

    // Collect coins (with magnet effect)
    for (const coin of coinList) {
      if (!coin.collected) {
        const coinSize = coin.size
        const dx = (player.x + PLAYER_SIZE / 2) - (coin.x + coinSize / 2)
        const dy = (player.y + PLAYER_SIZE / 2) - (coin.y + coinSize / 2)
        const dist = Math.sqrt(dx * dx + dy * dy)

        // Magnet effect - pull coins towards player
        if (player.hasMagnet && dist < MAGNET_RADIUS) {
          const pullSpeed = 8
          const angle = Math.atan2(dy, dx)
          coin.x += Math.cos(angle) * pullSpeed
          coin.y += Math.sin(angle) * pullSpeed
        }

        if (dist < PLAYER_SIZE / 2 + coinSize / 2) {
          coin.collected = true

          if (coin.type === 'doubleJump') {
            player.hasDoubleJump = true
          } else if (coin.type === 'magnet') {
            player.hasMagnet = true
            player.magnetEndTime = Date.now() + 30000 // 30 seconds
          } else if (coin.type === 'pillarDrop') {
            // Start pillar drop animation
            pillarDropping = true
            pillarDropProgress = 0
          } else {
            coins.value += coin.value
            store.coinsCollected = coins.value
          }
        }
      }
    }

    // Check if all coins collected or none left
    const hasVisibleCoins = coinList.some(c => !c.collected)
    if (!hasVisibleCoins) {
      spawnCoins()
    }

    // Check if player fell off
    if (player.y > height + 100) {
      falls.value++
      const randomPillar = pillars[Math.floor(Math.random() * pillars.length)]
      player.x = randomPillar.x + randomPillar.width / 2 - PLAYER_SIZE / 2
      player.y = -PLAYER_SIZE
      player.vy = 0
    }

  } else if (gameState === 'success') {
    transitionProgress += 0.02

    // Continue player movement controls during transition
    if (keys.left) player.x -= MOVE_SPEED
    if (keys.right) player.x += MOVE_SPEED
    player.x = Math.max(0, Math.min(canvas.value.width - PLAYER_SIZE, player.x))

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

  // Draw pillars
  for (const pillar of pillars) {
    // Draw pillar body
    ctx.fillStyle = '#DA291C'
    ctx.fillRect(pillar.x, pillar.y, pillar.width, height - pillar.y + 200)

    // Pillar studs protruding from top
    const studCols = pillar.isDoubleWidth ? 8 : 4
    const studSpacingX = pillar.width / studCols
    const studHeight = 3
    const studTopY = pillar.y - studHeight

    // Draw studs as cylinders
    for (let col = 0; col < studCols; col++) {
      const sx = pillar.x + studSpacingX * (col + 0.5)

      // Draw cylinder body (darker red)
      ctx.fillStyle = '#b8231a'
      ctx.fillRect(sx - STUD_SIZE * 1.5, studTopY, STUD_SIZE * 3, studHeight)

      // Draw top ellipse of cylinder (darker red)
      ctx.beginPath()
      ctx.ellipse(sx, studTopY, STUD_SIZE * 1.5, STUD_SIZE * 0.6, 0, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  // Draw coins (hide during success transition)
  if (gameState !== 'success') {
    for (const coin of coinList) {
      if (!coin.collected) {
        const coinSize = coin.size
        const cx = coin.x + coinSize / 2
        const cy = coin.y + coinSize / 2

        // Set color based on type
        if (coin.type === 'doubleJump') {
          ctx.fillStyle = '#00FF00' // Green
          ctx.strokeStyle = '#000'
        } else if (coin.type === 'magnet') {
          ctx.fillStyle = '#9932CC' // Purple
          ctx.strokeStyle = '#000'
        } else if (coin.type === 'pillarDrop') {
          ctx.fillStyle = '#FF0000' // Red
          ctx.strokeStyle = '#000'
        } else {
          ctx.fillStyle = '#FFD700' // Gold
          ctx.strokeStyle = '#000'
        }

        ctx.lineWidth = coin.type === 'big' ? 3 : 2
        ctx.beginPath()
        ctx.arc(cx, cy, coinSize / 2, 0, Math.PI * 2)
        ctx.fill()
        ctx.stroke()

        // Inner circle
        if (coin.type === 'doubleJump') {
          ctx.fillStyle = '#00CC00'
        } else if (coin.type === 'magnet') {
          ctx.fillStyle = '#7B28A8'
        } else if (coin.type === 'pillarDrop') {
          ctx.fillStyle = '#CC0000'
        } else {
          ctx.fillStyle = '#FFA500'
        }
        ctx.beginPath()
        ctx.arc(cx, cy, coinSize / 4, 0, Math.PI * 2)
        ctx.fill()
      }
    }
  }

  // Draw player (blue 2x2 brick) with fade during success
  const playerAlpha = gameState === 'success' ? (1 - transitionProgress) : 1
  ctx.globalAlpha = playerAlpha

  ctx.fillStyle = '#0055BF'
  ctx.fillRect(player.x, player.y, PLAYER_SIZE, PLAYER_SIZE)

  // Player studs facing forward (2x2 grid viewed from side)
  ctx.fillStyle = '#004494'
  const pStudSpacing = PLAYER_SIZE / 2

  // Draw 2x2 grid of circular studs, evenly distributed
  for (let row = 0; row < 2; row++) {
    for (let col = 0; col < 2; col++) {
      const sx = player.x + pStudSpacing * (col + 0.5)
      const sy = player.y + pStudSpacing * (row + 0.5)
      ctx.beginPath()
      ctx.arc(sx, sy, STUD_SIZE, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  // Border on player
  ctx.strokeStyle = '#003366'
  ctx.lineWidth = 2
  ctx.strokeRect(player.x, player.y, PLAYER_SIZE, PLAYER_SIZE)

  ctx.globalAlpha = 1
}

function gameLoop() {
  update()
  draw()
  animationId = requestAnimationFrame(gameLoop)
}

function handleKeyDown(e) {
  if (gameState !== 'playing' && gameState !== 'success') return

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
    // Just redraw, don't reinitialize game on resize
    canvas.value.width = container.value.clientWidth
    canvas.value.height = container.value.clientHeight
  }
}

onMounted(async () => {
  initGame()
  gameLoop()

  window.addEventListener('keydown', handleKeyDown)
  window.addEventListener('keyup', handleKeyUp)
  window.addEventListener('resize', handleResize)

  // If standalone mode, just play forever
  if (props.standalone) {
    gameState = 'playing'
    return
  }

  // Otherwise, wait for build promise
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

.falls {
  position: absolute;
  top: calc(var(--spacing-md) + 56px);
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

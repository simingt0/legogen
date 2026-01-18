import { reactive } from 'vue'

// Shared state across pages
export const store = reactive({
  // User inputs
  imageFile: null,
  imagePreviewUrl: null,
  description: '',

  // Build results
  buildResult: null,
  buildError: null,

  // Loading state
  isBuilding: false,
  buildPromise: null,

  // Mini-game stats
  coinsCollected: 0
})

// Actions
export function setImage(file) {
  store.imageFile = file
  if (file) {
    store.imagePreviewUrl = URL.createObjectURL(file)
  } else {
    store.imagePreviewUrl = null
  }
}

export function setDescription(desc) {
  store.description = desc
}

export function clearBuildResult() {
  store.buildResult = null
  store.buildError = null
}

export function resetForNewBuild() {
  store.buildResult = null
  store.buildError = null
  store.coinsCollected = 0
  store.isBuilding = false
  store.buildPromise = null
}

export async function startBuild() {
  if (!store.imageFile || !store.description) {
    throw new Error('Image and description required')
  }

  store.isBuilding = true
  store.buildResult = null
  store.buildError = null

  const formData = new FormData()
  formData.append('image', store.imageFile)
  formData.append('description', store.description)

  store.buildPromise = fetch('/build', {
    method: 'POST',
    body: formData
  })
    .then(async (response) => {
      const data = await response.json()
      if (data.success) {
        store.buildResult = data
        return data
      } else {
        throw new Error(data.error || 'Build failed')
      }
    })
    .catch((error) => {
      store.buildError = error.message
      throw error
    })
    .finally(() => {
      store.isBuilding = false
    })

  return store.buildPromise
}

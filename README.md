# LegoGen Setup & Usage Guide

This guide walks you through setting up, running, and using the project for Mac/Linux.

---

# Prerequisites

Before starting, make sure you have:

- Python 3.9+ (recommended)
- Node.js + npm
- `uv` (recommended Python package manager)

If you do not already have Node.js and npm installed, download and install Node.js from the official Node.js website. The standard Node.js installer includes npm.

Recommended: install the **LTS** version.

After installing, verify it worked:

```
node -v
npm -v
```

If you don't have `uv`:

```
pip install uv
```

# Setup
### Python
Create python environment (uv recommended) and install requirements.txt
```
cd path/to/legogen
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```
Now, you can start the backend:
```
cd server
uvicorn main:app --reload
```
### npm
Now, open a new terminal (cd to project directory) and run:
```
cd web
npm install
npm run dev
```
At this point you should see the frontend up and running on port 5173. If it does not automatically open, go to 'http://localhost:5173' in your browser.

# Usage
1. Upload image
2. Type prompt or select pre-loaded prompt
3. Play game while it generates (this can take a while if voxel size is big or if you are generating a new mesh)
4. View generated structure (⬅️➡️ keys to rotate view, space to change visualizer type)

### Cached Builds (do not require meshy API)
You can put any of these as text without a Meshy API key. 
- mushroom
- apple
- mug
- a donut
- pea shooter
- banana
- small house
- a formula one race car
- realistic rendering of dwayne the rock johnson's face
- small dragon
- a small house with a chimney
- sunglasses
- a lizard
- castle tower
- minecraft creeper
- a flower
- formula one race car

if you want to generate anything else, you will need a MESHY_API_KEY (~$20/mo)

### Tweaks
- to change the max dimension of the build (cube-shaped), change `voxel_size` on line 112 of server/main.py
- to allow builds with more bricks, run the server manually with `python main.py --bricks-json .path/to/bricks.json`, supplying the json file with the simulated bricks you want to use. you will still need to upload an image, but the bricks.json will override the bricks from the image.

### things to fix
- /build should reroute to root when not valid (ideally it saves + reloads the current build until you regenerate)
- having a clear cli for running with variations would be good, add cli args for
  - MESHY_MODE
  - voxel dimensions
  - bricks.json
  - putting all of these into a makefile/justfile would be awesome
- not requiring an image to be uploaded when bricks.json is set

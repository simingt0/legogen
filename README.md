# setup
### python
create python environment (uv recommended) and install requirements.txt
`uv venv`
`source .venv/bin/activate`
`uv pip install -r requirements.txt`
`cd server`
`uvicorn main:app --reload`

### npm
install packages and run dev server
`cd web`
`npm install`
`npm run dev`

at this point you should see the frontend up and running on port 5173.

# usage
1. upload image
2. type prompt or select pre-loaded prompt
3. play game while it generates
4. view generated structure

### cached builds (do not require meshy credits)
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

### tweaks
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

import jax
import jax.numpy as jnp
import numpy as np

def make_grid_visualization(vis, grid=4, max_bz=4):
  assert vis.ndim == 4
  n, h, w, c = vis.shape

  col = grid
  row = min(grid, n // col) 
  if n % (col * row) != 0:
    n = col * row * max_bz
    vis = vis[:n]
    n, h, w, c = vis.shape
  assert n % (col * row) == 0

  vis = vis.reshape((-1, col, row * h, w, c))
  vis = jnp.einsum('mlhwc->mhlwc', vis)
  vis = vis.reshape((-1, row * h, col * w, c))

  # `[:bz][0]` is just `[0]`; always returns one grid. max_bz governs how many
  # samples to keep when n doesn't fit a single grid (see above).
  return jax.device_get(vis[0])

def float_to_uint8(vis):
  if not isinstance(vis, np.ndarray):
    vis = jax.device_get(vis)
  vis = (vis + 0.5) * 255.0
  vis = np.clip(vis, 0, 255)
  vis = vis.astype(np.uint8)
  return vis

VIS_PROMPTS = [
  'endless book labyrinth illustration for The City of Dreaming Books by walter moers bookshelf dungeon glowing mushrooms scenic light under city underground rustic old place ar 169', 

  'beautiful Jaguar decorated with huichol beads, in the jungle, plants everywhere, DMT colours, ultra realistic , cinematic lighting   v 5', 

  'river of fruits and flowers exploding ', 

  'rain pouring down on a vintage car from 1950 era and trees are in the background of the car. Its a night shot 35 mm', 

  'a baby blue paper cup for bubble tea with whipped cream on top', 

  'gorgeous african queen gal standing by the window supercool swanky penthouse interior at night, amazing lighting, architectural digest photograph, brilliant colors, chaos, anarchy, liberty, independence, soul and afropop vibes, very detailed, photo taken with Hasselblad X1D, ISO 100, national geographic ', 

  'botanical drawing, psylocibe cubensis, realistic', 

  'Notorious B.I.G sitting slumped over a golden throne, with a kings crown, wearing a fur coat, at the back of the throne, a red color, as if he was in hell, color coded, ultradetailed, ultrarealistic, ultra high quality, ultra high definition, Careful composition, sharpen, insane details, cinematic lights, photorealism, 30mm shot Shutter Speed 1125, F5.6, White Balance, Megapixel, Pro Photo RGB, Unreal Engine, Cinematic, Chromatic Aberration, 8k, 4k ', 

  'Futuristic scifi city center, bustling crowds, tier 2 civilization, advanced technology, towering skyscrapers, neon lights, flying vehicles, inspired by the art of Syd Mead and the movie Blade Runner, vibrant and dynamic urban landscape ', 

  'Amazon Rainforest river ', 

  'Abtract painting, Dinner with lights is an abstract picture. Asian girl sitting ready to eat a glowing plate ', 

  'assortment of house plants in pots of various shapes and colors, placed on shelves ', 

  'fantastical microscopic fairy kingdom photographed by science lab, microscopic art, 8k, UHD ', 

  'a young African American fireman in worn out fireman gear, backlit against a dark background. Full body depiction. Hyper realistic, highly detailed, 8k, photo.', 

  'progressive renaissance medieval fortress city in spanish country side. Cinematic, Color Grading, Photography, Shot on 50mm lense, Ultra  Wide Angle, intricate details, beautifully color graded, Unreal Engine, Cinematic, Editorial Photography, Shot on 85mm lens, White Balance, Halfrear Lighting, Backlight, Natural Lighting, Cinematic Lighting, Studio Lighting, Global Illumination, Screen Space Global Illumination, Ray Tracing Global Illumination, Optics, Scattering, Ray Tracing Reflections, Lumen Reflections, Screen Space Reflections, Chromatic Aberration, Ray Traced, Ray Tracing Ambient Occlusion, Anti  Aliasing, FKAA, TXAA, RTX, SSAO, Shaders, OpenGL  Shaders, GLSL  Shaders, Post Processing, Post  Production, Tone Mapping, CGI, 4k, high detailed, ', 

  'a High Definition, cinematic movie still of Clint Eastwood using complicated DJ equipment while dressed like a raver on the beach at an all night rave party'
]

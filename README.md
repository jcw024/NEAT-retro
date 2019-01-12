# NEAT-retro
Self learning AI learns to play Space Invaders and Streets of Rage 2 using NEAT and OpenAI retro. NEAT (NeuroEvolution of Augmenting Toplogies) evolves a neural network over several generations of populations as in biological evolution. NEAT interfaces with the games using OpenAI Retro to collect pixel observations and rewards and apply button inputs to the environment.
### Installation
git clone https://github.com/jcw024/NEAT-retro.git

cd NEAT-retro

pip install -r requirements.txt

python -m retro.import SpaceInvaders/SpaceInvaders.sfc

python -m retro.import StreetsOfRage2/StreetsOfRage.md

you will also need to move everything in StreetsOfRage2/retro_files/ to a local copy of openai retro at retro/data/stable/StreetsOfRage2-Genesis 
### Youtube Links
Space Invaders (SNES): https://youtu.be/hZ02Lwfw4Ws

Streets of Rage 2 (Genesis): https://youtu.be/kYv_TiF5G1E
### References
NEAT: http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

OpenAI Retro: https://github.com/openai/retro

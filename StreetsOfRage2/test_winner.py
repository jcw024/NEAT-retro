#! /usr/bin/env python
import retro
import numpy as np
import cv2 
import neat
import pickle
import argparse

env = retro.make('StreetsOfRage2-Genesis', '1Player.Axel.Level1v2.state', record='.')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config_reduce5')

with open('winner_gen40_pop50.pkl','rb') as f:
    c = pickle.load(f)

ob = env.reset()
ac = env.action_space.sample()

inx, iny, inc = env.observation_space.shape

inx = int(inx/8)
iny = int(iny/8)
print(inx,iny)

net = neat.nn.recurrent.RecurrentNetwork.create(c, config)

current_max_fitness = 0
fitness_current = 0
frame = 0
counter = 0

done = False
cv2.namedWindow("main", cv2.WINDOW_NORMAL)

imgarray = []
while not done:
    
    env.render()
    frame += 1
    scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2RGB)
    scaledimg = cv2.resize(scaledimg, (iny, inx)) 
    scaledimg = np.delete(scaledimg, range(0,9), axis=0)
    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2RGB)
    ob = np.reshape(ob, (inx,iny,3))
    ob = np.delete(ob, range(0,9), axis=0)      #remove top few rows of pixels to increase speed
    cv2.imshow('main', scaledimg)
    cv2.waitKey(1) 

    imgarray = np.ndarray.flatten(ob)

    nnOutput = net.activate(imgarray)
    
    ob, rew, done, info = env.step(nnOutput)
    fitness_current += rew
    
    if fitness_current > current_max_fitness:
        current_max_fitness = fitness_current
        counter = 0
    else:
        counter += 1
    if done:
        done = True
        
    c.fitness = fitness_current
    #done=True

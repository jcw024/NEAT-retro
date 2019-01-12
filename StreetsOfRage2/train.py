#! /usr/bin/env python
from __future__ import division
import retro
import numpy as np
import cv2 
import neat
import pickle
import cProfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', '-c', type=str, help='checkpoint file to continue previous run')
args = parser.parse_args()

#trains single genome within generation
def eval_genomes(genome, config):
    ob = env.reset()
    ac = env.action_space.sample()

    inx, iny, inc = env.observation_space.shape
    inx = int(inx/8)
    iny = int(iny/8)

    net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
    
    current_max_fitness = 0
    fitness_current = 0
    counter = 0
    lives_tot = 3
    horiz_tot = 74
    vert_tot = 184
    
    #cv2.namedWindow("main", cv2.WINDOW_NORMAL)     #for visualizing input pixels
    done = False
    while not done:
        env.render()

        #scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2RGB)
        #scaledimg = cv2.resize(scaledimg, (iny, inx)) 
        #scaledimg = np.delete(scaledimg, range(0,9), axis=0)
        ob = cv2.resize(ob, (inx, iny))
        ob = cv2.cvtColor(ob, cv2.COLOR_BGR2RGB)    #include colors in observation
        ob = np.reshape(ob, (inx,iny,3))
        ob = np.delete(ob, range(0,9), axis=0)      #remove top few rows of pixels to increase speed
        #cv2.imshow('main', scaledimg)
        #cv2.waitKey(1) 
        
        imgarray = np.ndarray.flatten(ob)

        nnOutput = net.activate(imgarray)
        ob, rew1, done1, info = env.step(nnOutput)
        ob, rew2, done2, info = env.step(nnOutput)
        ob, rew3, done3, info = env.step(nnOutput)
        rew = (rew1 + rew2 + rew3)
        lives = info['lives']

        fitness_current += rew
        if fitness_current > current_max_fitness:
            current_max_fitness = fitness_current
            counter = 0
        else:
            counter += 1
        if any([done1, done2, done3]) or counter == 400:
            done = True
        genome.fitness = fitness_current
    print(genome.fitness)
    return genome.fitness
                
#setup training population parameters        
def main(checkpoint=None):
    if checkpoint is not None:
        p = neat.checkpoint.Checkpointer.restore_checkpoint(checkpoint)
    else:
        p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=5,time_interval_seconds=3600))

    pe = neat.ParallelEvaluator(5, eval_genomes)    #run on multiple cores
    winner = p.run(pe.evaluate, 40)                 #do training for 40 generations

    with open('winner_reduce5_rgb.pkl', 'wb') as output:
        print('writing winner gen to ', output)
        pickle.dump(winner, output)
    
if __name__ == '__main__':
    env = retro.make('StreetsOfRage2-Genesis', '1Player.Axel.Level1v2', scenario='custom_scenario') 
    imgarray = []
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config')
    main(args.checkpoint)

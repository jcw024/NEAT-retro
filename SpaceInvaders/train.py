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
    inx = int(inx/5)
    iny = int(iny/5)

    net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
    
    current_max_fitness = 0
    fitness_current = 0
    counter = 0
    lives_tot = 3
    
    #cv2.namedWindow("network_input", cv2.WINDOW_NORMAL)    #show input pixels to neat
    done = False
    while not done:
        env.render()

        #shrink screen for fewer pixel observations per loop
        #scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
        #scaledimg = cv2.resize(scaledimg, (iny, inx)) 
        ob = cv2.resize(ob, (inx, iny))
        ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
        ob = np.reshape(ob, (inx,iny))
        #cv2.imshow('network_input', scaledimg)
        #cv2.waitKey(1) 
        
        imgarray = np.ndarray.flatten(ob)

        nnOutput = net.activate(imgarray)
        ob, rew1, done1, info = env.step(nnOutput)  #3 steps to skip some frames
        ob, rew2, done2, info = env.step(nnOutput)
        ob, rew3, done3, info = env.step(nnOutput)
        rew = (rew1 + rew2 + rew3)
        lives = info['lives']
        if lives < lives_tot:
            #fitness_current -= 100
            lives_tot = lives
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
    p.add_reporter(neat.Checkpointer(generation_interval=10,time_interval_seconds=3600))

    pe = neat.ParallelEvaluator(5, eval_genomes)    #run on multiple cores
    winner = p.run(pe.evaluate, 40)                 #do training for 40 generations

    with open('winner_pop50_gen40.pkl', 'wb') as output:
        print('writing winner gen to ', output)
        pickle.dump(winner, output)
    
if __name__ == '__main__':
    env = retro.make('SpaceInvaders-Snes', '1Player.ClassicMode.UprightCabinet.state') 
    imgarray = []
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config')
    main(args.checkpoint)

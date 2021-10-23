#!/usr/bin/env python3.5 
from mchgenalg import GeneticAlgorithm
import mchgenalg
import numpy as np
import os
import time
start_time = time.time()

timesEvaluated = 0
bestepochs = -1

#remove log files
#tracks how many times GA fitness function has been invoked
if os.path.exists("logs_fitness_function_invoked.txt"):
  os.remove("logs_fitness_function_invoked.txt")

#tracks success rate for each action (all steps throughout the program execution)
if os.path.exists("logs_success_rate.txt"):
  os.remove("logs_success_rate.txt")

#logs reward for all actions
if os.path.exists("rewards.csv"):
  os.remove("rewards.csv")

#saves epoch value for success rate > threshold (train.py)
if os.path.exists("epochs.txt"):
  os.remove("epochs.txt")

#logs general logging comments
if os.path.exists("logs_common.txt"):
  os.remove("logs_common.txt")

#logs success rate after rollout workers complete (each epoch)
if os.path.exists("logs_success_rate_per_epoch.txt"):
  os.remove("logs_success_rate_per_epoch.txt")

#logs success being set in rollout.py
if os.path.exists(
        "logs_common_is_success.txt"):
  os.remove("logs_common_is_success.txt")

#delete logs folder
if os.path.exists("/tmp/openaiGA"):
  os.remove("/tmp/openaiGA")

# First, define function that will be used to evaluate the fitness
def fitness_function(genome):
    
    global timesEvaluated
    timesEvaluated += 1
    start_time = time.time()
    with open('logs_fitness_function_invoked.txt', 'a') as output:
        output.write(str(timesEvaluated) + "\n")
    print("Fitness function invoked "+str(timesEvaluated)+" times")

    #setting parameter values using genome
    polyak = decode_function(genome[0:10])
    if polyak >= 1:
        polyak = 0.999 #1
    gamma = decode_function(genome[11:21])
    if gamma >= 1:
        gamma = 0.999 #1
    Q_lr = decode_function(genome[22:33])
    # if Q_lr >= 1:
    #     Q_lr = 0.999 #1
    Q_lr = 0.001
    pi_lr = decode_function(genome[34:44])
    # if pi_lr >= 1:
    #     pi_lr = 0.999 #1
    pi_lr = 0.001
    random_eps = decode_function(genome[45:55])
    if random_eps >= 1:
        random_eps = 0.999 #1
    noise_eps = decode_function(genome[56:66])
    if noise_eps >= 1:
        noise_eps = 0.999 #1
    epochs_default = 20 #50
    env = 'AuboReach-v2' #'AuboReach-v0'
    logdir = '$HOME/openaiGA' #'/tmp/openaiGA'
    num_cpu = 6

    with open('logs_common.txt', 'a') as output:
        output.write("======Setting Parameters value========="+ "\n")
        output.write("Tau = " + str(polyak))
        output.write(" || Gamma = " + str(gamma))
        output.write(" || Q_learning = " + str(Q_lr))
        output.write(" || pi_learning = " + str(pi_lr))
        output.write(" || random_epsilon = " + str(random_eps))
        output.write(" || noise_epsilon = " + str(noise_eps) + "\n")

    query = "python3 -m train --env="+env+" --logdir="+logdir+" --n_epochs="+str(epochs_default)+" --num_cpu="+str(num_cpu) + " --polyak_value="+ str(polyak) + " --gamma_value=" + str(gamma) + " --q_learning=" + str(Q_lr) + " --pi_learning=" + str(pi_lr) + " --random_epsilon=" + str(random_eps) + " --noise_epsilon=" + str(noise_eps)

    print(query)
    #calling training to calculate number of epochs required to reach close to maximum success rate
    os.system(query)
    #epochs = train.launch(env, logdir, epochs_default, num_cpu, 0, 'future', 5, 1, polyak, gamma)
    #env, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return   

    ##tracking time to execute one run
    programExecutionTime = time.time() - start_time  # seconds
    programExecutionTime = programExecutionTime / (60)  # minutes
    with open('logs_common.txt', 'a') as output:
        output.write("======Run " + str(timesEvaluated) + " took " + str(programExecutionTime) + " minutes to complete=========" + "\n")

    file = open('epochs.txt', 'r')

    #one run is expected to converge before epochs_efault
    #if it does not converge, either add condition here, or make number of epochs as dynamic

    epochs = int(file.read())

    if epochs == None:
        epochs = epochs_default

    global bestepochs
    if bestepochs == -1:
        bestepochs = epochs
    if epochs < bestepochs:
        bestepochs = epochs
        with open('BestParameters.txt', 'a') as output:
            output.write("Epochs taken to converge : " + str(bestepochs) + "\n")
            output.write("Tau = " + str(polyak) + "\n")
            output.write("Gamma = " + str(gamma) + "\n")
            output.write("Q_learning = " + str(Q_lr) + "\n")
            output.write("pi_learning = " + str(pi_lr) + "\n")
            output.write("random_epsilon = " + str(random_eps) + "\n")
            output.write("noise_epsilon = " + str(noise_eps) + "\n")
            output.write("\n")
            output.write("=================================================")
            output.write("\n")

    print('EPOCHS taken to converge:' + str(epochs))

    print("Best epochs so far : "+str(bestepochs))
    return 1/epochs

def decode_function(genome_partial):

    prod = 0
    for i,e in reversed(list(enumerate(genome_partial))):
        if e == False:
            prod += 0
        else:
            prod += 2**abs(i-len(genome_partial)+1)
    return prod/1000

# Configure the algorithm:
population_size = 50 #30
genome_length = 66
ga = GeneticAlgorithm(fitness_function)
ga.generate_binary_population(size=population_size, genome_length=genome_length)

# How many pairs of individuals should be picked to mate
ga.number_of_pairs = 5

# Selective pressure from interval [1.0, 2.0]
# the lower value, the less will the fitness play role
ga.selective_pressure = 1.5
ga.mutation_rate = 0.1

# If two parents have the same genotype, ignore them and generate TWO random parents
# This helps preventing premature convergence
ga.allow_random_parent = True # default True
# Use single point crossover instead of uniform crossover
ga.single_point_cross_over = False # default False

# Run 100 iteration of the algorithm
# You can call the method several times and adjust some parameters
# (e.g. number_of_pairs, selective_pressure, mutation_rate,
# allow_random_parent, single_point_cross_over)
ga.run(50) #30 default 1000
best_genome, best_fitness = ga.get_best_genome()

print("BEST CHROMOSOME IS")
print(best_genome)
print("It's decoded value is")
print("Tau = " + str(decode_function(best_genome[0:10])))
print("Gamma = " + str(decode_function(best_genome[11:22])))
print("Q_learning = " + str(decode_function(best_genome[23:33])))
print("pi_learning = " + str(decode_function(best_genome[34:44])))
print("random_epsilon = " + str(decode_function(best_genome[45:55])))
print("noise_epsilon = " + str(decode_function(best_genome[56:66])))

# If you want, you can have a look at the population:
population = ga.population

# and the fitness of each element:
fitness_vector = ga.get_fitness_vector()

# time tracking
programExecutionTime = time.time() - start_time #seconds
programExecutionTime = programExecutionTime/(60*60) #hours
with open('logs_common.txt', 'a') as output:
    output.write("======Total program execution time is " + str(programExecutionTime) +" hours=========" + "\n")
print("--- %s seconds ---" % (time.time() - start_time))

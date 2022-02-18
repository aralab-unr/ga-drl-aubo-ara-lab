import os
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns; sns.set()
import glob2
import argparse
import glob
import pandas as pd


def smooth_reward_curve(x, y):
    halfwidth = int(np.ceil(len(x) / 60))  # Halfwidth of our smoothing convolution
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
        mode='same')
    return xsmoo, ysmoo


def load_results(file):
    if not os.path.exists(file):
        return None
    with open(file, 'r') as f:
        lines = [line for line in f]
    if len(lines) < 2:
        return None
    keys = [name.strip() for name in lines[0].split(',')]
    data = np.genfromtxt(file, delimiter=',', skip_header=1, filling_values=0.)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.ndim == 2
    assert data.shape[-1] == len(keys)
    result = {}
    for idx, key in enumerate(keys):
        result[key] = data[:, idx]
    return result


def pad(xs, value=np.nan):
    maxlen = np.max([len(x) for x in xs])

    padded_xs = []
    for x in xs:
        if x.shape[0] >= maxlen:
            padded_xs.append(x)

        padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
        x_padded = np.concatenate([x, padding], axis=0)
        assert x_padded.shape[1:] == x.shape[1:]
        assert x_padded.shape[0] == maxlen
        padded_xs.append(x_padded)
    return np.array(padded_xs)


parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
#parser.add_argument('dir2', type=str)
parser.add_argument('--smooth', type=int, default=1)
args = parser.parse_args()
'''
##################uncomment so that you can get the graph for epochs vs function call ##################

four_count = 1
file_path = 'ga_success.csv'
epoch_list = []
df = pd.read_csv(file_path)
idx = 0
function_call = []
for i in range(len(df['epoch'])):
    four_count += 1
    if four_count % 4 == 0:
        epoch_list.append(df['epoch'][i])
        function_call.append(idx)
        idx += 1
        four_count = 0
        if idx == 200:
            break

plt.clf()

plt.plot(function_call, epoch_list)
plt.plot()
plt.title('Epochs to converge vs Times GA fitness function evaluated')
plt.xlabel('Times GA fitness function evaluated')
plt.ylabel('Epochs to converge')
plt.grid(False)
plt.legend()
#plt.savefig(os.path.join(args.dir, 'Aubo_Reach_Averages'))
plt.savefig('fetch_reach_ga')
'''




'''

##############uncomment to be able to get a graph that compare different parameter value runs ##################################
# Load all data.
epochs = 60
averages = np.zeros(epochs)
epoch_list = np.zeros(epochs)
averages2 = np.zeros(epochs)

csv_files = glob.glob(os.path.join(args.dir, "*.csv"))
csv_files2 = glob.glob(os.path.join(args.dir2, '*.csv'))
idx = 0
file_count = 0
# loop over the list of csv files
for f in csv_files:
    file_count += 1  
    # read the csv file
    df = pd.read_csv(f)
    print(df)
    length = len(df['test/success_rate'])
    print(length)
    for i in range(epochs):
        averages[i] += df['test/success_rate'][i]
        epoch_list[i] = df['epoch'][i]

final_average_list = averages/10

print(final_average_list.shape)
print(epoch_list.shape)

for f in csv_files2:
    df = pd.read_csv(f)
    for i in range(epochs):
        averages2[i] += df['test/success_rate'][i]
    
final_average_list2 = averages2/10

plt.clf()
x, y = smooth_reward_curve(epoch_list, final_average_list)
x2, y2 = smooth_reward_curve(epoch_list, final_average_list2)


plt.plot(x2, y2, label='DRL')
plt.plot(x, y, label='GA-DRL')
plt.plot()
plt.title('Success rate vs epochs')
plt.xlabel('epochs')
plt.ylabel('Median success rate')
plt.grid(False)
plt.legend()
plt.savefig(os.path.join(args.dir, 'Aubo_Reach_Averages'))

'''
'''

################ uncomment to print out the median rewards vs function call graph ##############################
import statistics
data = {}
epochs = 60
averages = np.zeros(epochs)
epoch_list = np.zeros(epochs)
averages2 = np.zeros(epochs)
reward_list = []
list_of_files = filter( os.path.isfile,
                        glob.glob(args.dir + '/' + '*') )
list_of_files = sorted(list_of_files,
                        key = os.path.getmtime)

idx = 0
file_count = 0
function_call = []
epoch_data = []
total_reward = 0
median_list = []
list1 = []
# loop over the list of csv files
for f in list_of_files:
    file_count += 1  
    # read the csv file
    df = pd.read_csv(f)
    #print(df)
    print(f)
    for i in range(len(df['test/success_rate'])):
        list1.append(df['test/success_rate'][i])

    median_list.append(statistics.median(list1))
    list1 = []
    function_call.append(idx)
    total_reward = 0
    idx += 1
    if idx == 200:
        break

print(median_list)
print(function_call)

plt.clf()


#plt.plot(x2, y2, label='DRL')
plt.plot(function_call, median_list)
plt.plot()
plt.title('Median success rate vs Times GA fitness function evaluated')
plt.xlabel('Times GA fitness function evaluated')
plt.ylabel('Median success rate')
plt.grid(False)
plt.legend()
plt.savefig(os.path.join(args.dir, 'success_vs_function2'))

'''
################ uncomment to print out the median rewards vs function call graph ##############################
import statistics
data = {}
epochs = 60
averages = np.zeros(epochs)
epoch_list = np.zeros(epochs)
averages2 = np.zeros(epochs)
reward_list = []
list_of_files = filter( os.path.isfile,
                        glob.glob(args.dir + '/' + '*') )
list_of_files = sorted(list_of_files,
                        key = os.path.getmtime)

idx = 0
file_count = 0
function_call = []
epoch_data = []
total_reward = 0
median_list = []
list1 = []
# loop over the list of csv files
for f in list_of_files:
    file_count += 1  
    # read the csv file
    df = pd.read_csv(f)
    #print(df)
    print(f)
    for i in range(len(df['test/reward'])):
        list1.append(df['test/reward'][i])

    median_list.append(statistics.median(list1))
    list1 = []
    function_call.append(idx)
    total_reward = 0
    idx += 1
    if idx == 200:
        break

print(median_list)
print(function_call)

plt.clf()


#plt.plot(x2, y2, label='DRL')
plt.plot(function_call, median_list)
plt.plot()
plt.title('Median reward vs Times GA fitness function evaluated')
plt.xlabel('Times GA fitness function evaluated')
plt.ylabel('Median reward')
plt.grid(False)
plt.legend()
plt.savefig(os.path.join(args.dir, 'reward_vs_function2'))

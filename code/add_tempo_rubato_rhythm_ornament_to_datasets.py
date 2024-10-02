"""
Author: Morgan Rivers
Date: Sept 7, 2024

This script augments the existing dataset with the categories of "tempo", "rythm", and "ornament",
which have been identified in the Nature paper "Contextual and combinatorial structure in sperm whale vocalisations"
as a way to make it easy for the training of transformers and the application of other techniques in
language decoding. Rubato is added in the generate_whale_dialogue_txt_with_proper_timings.py file.

See construct_string function comment below for more detail on the scheme used in this script.
"""

import pandas as pd
from numpy import genfromtxt
import pickle
import numpy as np


# Load the data
my_data = genfromtxt('../data/sperm-whale-dialogues.csv', delimiter=',', dtype=None, encoding="utf8")

# durations of whale codas
durs = my_data[1:,2].astype(float)

# names of whale conversations (conversations between whales or from one whale happening at similar time)
file_names = my_data[1:,0]

# Load extra_clicks binary (aka ornaments). 1 if ornament was added or not the coda, 0 otherwise
extra_click = pickle.load(open('../data/ornaments.p', "rb"))

# the number of clicks in the coda
num_clicks = my_data[1:,1].astype(int)

# the starting time stamp of the coda
TsTo = my_data[1:,-1].astype(float)

# the rhythm category stamp of the coda (0-17)
rhythms = pickle.load(open("../data/rhythms.p","rb"))


# Define the return_tempo function, gets the tempo category of the coda based on coda duration.
# Tempo is just a measure of how long it took for a coda to be vocalized.
def return_tempo(dur):
    if dur < 0.45:
        return 0
    elif dur < 0.61:
        return 1
    elif dur < 0.93:
        return 2
    elif dur < 1.08:
        return 3
    else:
        return 4


# Create the data
data = {
    'Whale': [my_data[i,-2].astype(int) for i in range(1, my_data.shape[0])],
    'File': [my_data[i,0][:6] for i in range(1, my_data.shape[0])],
    'Time': [my_data[i,-1].astype(float) for i in range(1, my_data.shape[0])],
    'Tempo': [return_tempo(my_data[i,2].astype(float)) for i in range(1, my_data.shape[0])],
    'Rhythm': [rhythms[i-1] for i in range(1, my_data.shape[0])],
    'Clicks': [my_data[i,1].astype(int) for i in range(1, my_data.shape[0])],
    'Extra Click': [extra_click[i-1] for i in range(1, my_data.shape[0])]
}



def interrupted(i):
    """
    If coda i was interrupted by the next whale coda, return the index of the next (interrupting) coda.
    Otherwise return -1.
    """
    # Start and end times of the click
    st = TsTo[i]
    en = TsTo[i]+durs[i]

    # if it's the last item, return -1. The last coda can't be interrupted by the next one.
    if i==my_data.shape[0]-1: # only check the previous
        return -1
    else:
        #the index of the next coda
        nex = i+1

        # return if next index is the last item
        if nex>=3840:
            return -1

        elif TsTo[nex]<=en and TsTo[nex]>=st:
            # if the starting timestamp of the next whale vocalization started before the end of this timestamp
            # of the current one, and the starting timestamp of the next vocalization is also starting after this one began.
            return nex
        else:
            return -1


# Create the DataFrame
df = pd.DataFrame(data)

# Define a function to help create the constructed string which helps us visualize the type of each coda
def construct_string(data):
    """
    Here's how the string is constructed:
    - There are 17 discernable rythms whales use for codas. We use the letters A-R, a different letter for each
      different rhythm of the codas.
    - The letter representing the coda is capitalized if it's ornamented, which just means it has an extra click.
      Otherwise its lowercase.
    - If the coda is interrupting a previous coda, the relative duration of this second coda defines its rubato.
      Most coda durations are constant (-), but sometimes the duration reduces (\) and sometimes it increases (/).
      Non-interrupting codas do not have rubatos appended. NOTE: Rubato is only added in generate_whale_dialogue_txt_with_proper_timings.py

    For example, the coda rythm category 13, which is typically labelled 5R3 (just making this up as an example, that's probably wrong)
    might get a letter "h" or "H". If it's not ornamented, then we would have it be "h" as the lowercase. Then, we
    Append a number between 1 and 5 for the tempo category, 1 if slow, 5 if very fast. Let's say it's in the middle at 3.
    And finally if this is interrupting a previous whale vocalization, but it has a statistically significant increased tempo
    than the other, it would be the "/" rubato tag appended, leaving us with the "word" spelled like:
    "/h3"
    All codas are represented with between 2 and three characters depending on if they have rubato.
    """
    rhythm = chr(ord('a') + data['Rhythm']).upper() if data['Extra Click'] == 1 else chr(ord('a') + data['Rhythm'])
    return rhythm + str(data['Tempo'] + 1)

# Create new column 'ConstructedString'
df['ConstructedString'] = df.apply(construct_string, axis=1)



# Combine and export the data

transposed_array = np.array(my_data[1:]).T
column_names = my_data[0]
original_columns = {}
for i in range(len(column_names)):
    column_name = column_names[i]
    original_columns[column_name] = transposed_array[i]

# Save the DataFrame df as a csv file
original_columns_df = pd.DataFrame(original_columns)

df = pd.concat([original_columns_df, df], axis=1)

df.to_csv('../data/sperm-whale-dialogues_augmented.csv', index=False)
print("The original data")
print(original_columns_df.head())
print()
print("New categorization columns added to data:")
print(df.head())

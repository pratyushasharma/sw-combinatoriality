"""
This python script generates a dialogue using the augmented whale csv data. It does so by first labelling the
whale by its number ("Whale" column) and the words that it says. When another whale speaks, the constructedText
column is concatenated until another whale speaks or the conversation ends. New conversations, defined by having
a different "filename", are separated and labelled by their filename.

This also includes proper timings: printing the time of the new conversations, as well as reordering the display by
timestamp and printing when codas are spoken effectively simultaneously.

"""
import pandas as pd
import numpy as np
import os
from numpy import genfromtxt
import pickle
from IPython import embed
from scipy.io import loadmat

# Load the data
data = pd.read_csv('../data/sperm-whale-dialogues_augmented.csv')

# Parses, cleans and orders the annotations in the order of appearence of the clicks

# Initialize all the different conversations as empty arrays
annotations = {}
for i, row in data.iterrows():
    # If it's the start of a new conversation, add a new dialogue
    file_name = row['File'][:9]
    annotations[file_name] = []

# Then we go over the entries of the annotation file one by one (row wise) and append the annotation to the list corresponding to the file name
# The annotations are not ordered in time in the annotation file so it is a random order.

for i, row in data.iterrows():
    file_name = row['File'][:9]
    time_start = float(row['TsTo']) # start time of the first click of the coda seq

    # collect all files with that name
    temp = [] # List of times the clicks of the codas lie at - (t1-t1),(t2-t1),(t3-t1)...
    temp.append(0)
    j = 3
    while (float(row.iloc[j]) != 0 and j<31): # Until the value of the ICI is zero we continue to append (tn-t1)
        if float(row.iloc[j])> 0.0002: # To ignore mistakes in annotations
            temp.append(0+np.sum(row.iloc[3:(j+1)].astype(float))) # t3-t1 = (t3-t2)+(t2-t1) = ICI2+ICI1
        j = j+1
    annotations[file_name].append([temp, time_start, int(row['Whale']), row['ConstructedString'], 0]) # Coda seq, what_time, what_whale_number



# Annotation is now a dict of annotations but is still not ordered in time
# Convert the dict of lists into a list of numpy array
books = []
for key in annotations.keys():
    story = annotations[key] #each entity is a line of a coda


    book = np.zeros((len(story),44))
    words = [''] * len(story) # store all the "words" in the story.
    for j in range(len(story)):
        book[j,0] = story[j][2] # Who made the sound (ID number)
        book[j,1] = story[j][1] # What time did it start at
        book[j,2:len(story[j][0])+2] = story[j][0] # Coda sequence
        book[j,43] = story[j][-1] # Labels
        words[j] = story[j][3] # The string sequence corresponding to this word
    books.append((book,np.array(words),key))


# Sorts the annotation arrays of each audio file in order of time
edited = []
for i in range(len(books)):
    book = books[i][0]
    words = books[i][1]
    key = books[i][2]


    # both the words and the books are sorted by the time
    sorted_book = book[np.argsort(book[:, 1])]
    sorted_words = words[np.argsort(book[:, 1])]

    edited.append((sorted_book,sorted_words,key))

# CODA grouping parameters:
max_diff = 10 # Max time difference otherwise print that there was a pause

# visualization parameters:
average_ICI = 0.2
duration_coda = 8*average_ICI # seconds for the vertical axis



previous_audiobook = ''
page_time_origin = -10000000000


newfile=0
starting_book =0
rootname='sw061b'
# duration_page = 60*3 # seconds for the horizontal axis

def determine_rubato(word_string_previous, word_string, click_times_previous, click_times,t_diff):
    if t_diff > 10:
        return " "

    assert word_string[0].lower() in ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r']
    assert word_string_previous[0].lower() in ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r']

    if word_string[0].isupper():
        # Ignore click if just for ornament
        coda_duration_without_ornament = click_times[-2]
    else:
        coda_duration_without_ornament = click_times[-1]

    if word_string_previous[0].isupper():
        # Ignore click if just for ornament
        coda_duration_without_ornament_previous = click_times_previous[-2]
    else:
        coda_duration_without_ornament_previous = click_times_previous[-1]



    rhythm_previous = word_string_previous.lower()[0]
    rhythm = word_string.lower()[0]
    tempo_previous = word_string_previous.lower()[1]
    tempo = word_string.lower()[1]

    # the supplement to the paper defines rubato as change of duration within the same tempo and rhythm class
    # however, a histogram statistical plot of the duration delta does not show any sort of bimodal results when
    # these return " ". Also, the durations tend to match between codas regardless of rhythm is a key result of
    # the paper. So it would maybe make sense to remove these returns. I guess tempo category would convey the appropiate
    # Information regardless though.
    if rhythm_previous != rhythm:
        return " "

    if tempo_previous != tempo:
        return " "

    duration_delta = coda_duration_without_ornament - coda_duration_without_ornament_previous

    return duration_delta

def categorize_rubato(rubato):
    # quantile_25th = np.percentile(numeric_rubato_deltas, 25)
    quantile_25th = -0.021416925000000087
    # quantile_75th = np.percentile(numeric_rubato_deltas, 75)
    quantile_75th = 0.018462550000000105

    # Assign categories based on these quantiles
    if rubato < quantile_25th:
        # Decreasing
        return str("\\")
    elif rubato < quantile_75th:
        # constant
        return str("-")
    else:
        # increasing
        return str("/")

# Change this value depending on how many audiofiles you want to see plotted (one audio file may be split into several pages)
num_audiobooks = len(edited) # max value it can take = len(edited)

# Initialize an empty list to hold the dialogues
dialogues = []
# with open('../data/whale_dialogues_with_choruses.txt', 'w') as f:
rubato_deltas = []
for idx in range(num_audiobooks):
    book = edited[idx][0]
    words = edited[idx][1]
    name = edited[idx][2]
    rootname = name[0:6]
    # print("----------------------------------------------------------------")
    # print('idx: '+str(idx)+' File:'+name)
    # print('   number of codas='+str(book.shape[0]-1))
    # print('  '+ rootname +' previous=' + previous_audiobook)
    # print("----------------------------------------------------------------")
    # f.write(f"\nFile: {name}\n")


    dialogues.append({
        'file': name,
        'dialogue': []
    })

    # Function: parse de coda and get basic info
    def parseCoda(i):
        if i==-1:
            return [0,0,0,0,0,0]
        coda = book[i,:]
        word_string = words[i]
        whale_number = coda[0].astype(int)
        t_init = coda[1]-time_origin
        click_times = coda[2:42].astype(float).tolist()
        num_clicks = np.count_nonzero(click_times)+1 # sum 1 because the first click is always zero.
        click_times = click_times[0:num_clicks]
        label_ = coda[43].astype(int)
        return [whale_number,t_init,word_string,num_clicks,click_times,label_] #,average_power,click_power]

    # Function: returns the index for the next coda (after coda i) from the same whale.
    def getNextCoda(i, whale_number):
        found = -1
        while found==-1 and i<book.shape[0]-1:
            i=i+1
            [whale_number_i,t_init,word_string,num_clicks,click_times,label_] = parseCoda(i)
            if whale_number_i==whale_number:
                found = i
        return found

    # Function: returns the index for the previous coda (before coda i) from the same whale.
    def getPreviousCoda(i, whale_number):
        found = -1
        while found==-1 and i>0:
            i=i-1
            [whale_number_i,t_init,word_string,num_clicks,click_times,label_] = parseCoda(i)
            if whale_number_i==whale_number:
                found = i
        return found


    if rootname == previous_audiobook:
        page_time_origin = -10000000000
    else:
        time_origin = book[0,1]
        previous_audiobook = rootname
        newfile=1

    for i in range(book.shape[0]):

        [whale_number,t_init,word_string,num_clicks,click_times,label_curr] = parseCoda(i)
        i_next = getNextCoda(i, whale_number)
        [whale_number_next, t_init_next, word_string_next, num_clicks_next, click_times_next,label_next] = parseCoda(i_next)
        i_previous = getPreviousCoda(i, whale_number)
        [whale_number_previous, t_init_previous, word_string_previous, num_clicks_previous, click_times_previous,label_previous] = parseCoda(i_previous)
        rubato_string = " "
        if i_previous != -1: # not the first item
            t_diff = t_init - t_init_previous
            rubato = determine_rubato(word_string_previous, word_string, click_times_previous, click_times,t_diff)
            if rubato != " ":
                rubato_deltas.append(rubato)
                rubato_string = categorize_rubato(rubato)
        dialogues[-1]['dialogue'].append({
            'whale': whale_number,
            'text': rubato_string+word_string, # the rubato comes between previous coda and this one, so put before.
            'timestamp': t_init
        })

""" Uncomment block below to see how rubato percentiles we calculated.
def plot_histogram_and_print_percentiles(rubato_deltas):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(rubato_deltas, bins=300, color='blue')  # Adjust number of bins as needed
    plt.title('Histogram of Rubato Deltas')
    plt.xlabel('Rubato Delta')
    plt.ylabel('Frequency')
    plt.show()

    # Calculate the 25th and 75th quantiles
    quantile_25th = np.percentile(rubato_deltas, 25)
    quantile_75th = np.percentile(rubato_deltas, 75)

    # Separate data based on quantiles
    data_decreasing = [delta for delta in rubato_deltas if delta < quantile_25th]
    data_constant = [delta for delta in rubato_deltas if quantile_25th <= delta <= quantile_75th]
    data_increasing = [delta for delta in rubato_deltas if delta > quantile_75th]

    # Prepare bins to be consistent across histograms
    bins = np.histogram_bin_edges(rubato_deltas, bins=100)  # Define bins from the entire data range

    # Plot three histograms and capture the bin counts
    fig, ax = plt.subplots()
    counts_decreasing, _, _ = ax.hist(data_decreasing, bins=bins, color='red', alpha=0.5, label='Decreasing')
    counts_constant, _, _ = ax.hist(data_constant, bins=bins, color='green', alpha=0.5, label='Constant')
    counts_increasing, _, _ = ax.hist(data_increasing, bins=bins, color='blue', alpha=0.5, label='Increasing')

    # Determine the maximum y-value across all histograms for consistent y-axis
    max_count = max(max(counts_decreasing), max(counts_constant), max(counts_increasing))
    ax.set_ylim([0, max_count])  # Set the y-axis limit to the maximum count

    # Add titles and labels
    ax.set_title('Histogram of Rubato Deltas with Quantile Ranges')
    ax.set_xlabel('Rubato Delta')
    ax.set_ylabel('Frequency')

    # Add legend
    ax.legend()

    # Show plot
    plt.show()

    print("quantile_25th")
    print(quantile_25th)
    print("quantile_75th")
    print(quantile_75th)
plot_histogram_and_print_percentiles(rubato_deltas)
"""

def print_chorus(chorus_whales_data, f):
    sorted_keys = sorted(chorus_whales_data)  # Sort the keys of the dictionary
    sorted_texts = [chorus_whales_data[key] for key in sorted_keys]  # Extract values in the sorted order of keys
    chorus_string = f"In chorus, whales {', '.join(map(str, sorted_keys))}: {' '.join(sorted_texts)}."
    f.write(chorus_string + "\n")

def print_time_no_vocalizations(time_diff,f):
    if time_diff < 60:
        # Less than a minute
        rounded = 5*(time_diff//5)
        unit_label = "second"
    elif time_diff < 3600:
        # Less than an hour
        units = time_diff // 60  # Convert to minutes
        if units < 5:
            rounded = units  # Keep exact if less than 5 minutes
        else:
            rounded = 5 * round(units / 5)
        unit_label = "minute"
    elif time_diff < 86400:
        # Less than a day, but more than an hour
        units = time_diff // 3600  # Convert to hours
        if units < 5:
            rounded = units  # Keep exact if less than 5 hours
        else:
            rounded = 5 * round(units / 5)
        unit_label = "hour"
    else:
        # One day or more
        units = time_diff // 86400  # Convert to days
        if units < 5:
            rounded = units  # Keep exact if less than 5 days
        else:
            rounded = 5 * round(units / 5)
        unit_label = "day"
    # Ensuring pluralization is correct based on the rounded number
    unit_label += "" if rounded == 1 else "s"
    f.write(f"\n(No vocalizations, {int(rounded)} {unit_label})\n\n")

# Open the output file
with open('../data/whale_dialogues.txt', 'w') as f:

    # Initialize a variable outside the loop to track the silent time
    # time_diff = 0

    # Print the dialogues
    for dialogue in dialogues:

        # Initialize variables
        previous_whale_name = ""  # empty as there is no previous, it's the start of a conversation
        what_last_whale_said = ""
        what_last_whale_said_array = []
        last_timestamp = -np.inf
        chorus_texts = {}

        # Write filename right above the start of the conversation to the file
        f.write(f"File: {dialogue['file']}\n")

        # Before starting the dialogue loop
        chorus_whales_data = {}
        chorus_text = ""
        previously_in_chorus = False
        previous_whale_utterance = ""

        # Inside the dialogue loop
        for i in range(len(dialogue['dialogue'])):
            line = dialogue['dialogue'][i]

            this_timestamp = line['timestamp']
            assert last_timestamp <= this_timestamp

            # Check time difference and manage chorus
            time_diff = abs(this_timestamp - last_timestamp)

            # uncomment below to print the line as well for debugging
            # f.write("\n"+str(line)+f" tdiff: {time_diff}\n")

            if time_diff < 0.5 and previous_whale_name != line['whale']:
                # if previous_whale_name and what_last_whale_said: # if not on the first one
                if not previously_in_chorus:
                    # we don't need to repeat this one, it will be in the chorus, so go up to the penultamate entry
                    if len(what_last_whale_said_array) > 1:
                        # If there was any previous stored thing to say, print it.
                        # Also remove the most previous as it's part of the chorus
                        f.write(f"Whale {previous_whale_name}: " + " ".join(what_last_whale_said_array[:-1]) + ".\n")

                # Add current whale to chorus if not already in it
                if previous_whale_name not in chorus_whales_data.keys():
                    chorus_whales_data[previous_whale_name] = previous_whale_utterance
                if line['whale'] not in chorus_whales_data.keys():
                    chorus_whales_data[line['whale']] = line['text']

                what_last_whale_said = ""
                what_last_whale_said_array = []
                previously_in_chorus = True

            else:
                # Output chorus if it exists and reset
                if previously_in_chorus: # if was chorus last time, and not this time.
                    print_chorus(chorus_whales_data, f)
                    chorus_whales_data = {}
                    what_last_whale_said = f"Whale {line['whale']}: {line['text']}"
                    what_last_whale_said_array.append(line['text'])
                else:
                    # Continue with the regular logic
                    if line['whale'] == previous_whale_name:
                        # We know the tdiff was large last time, but that might have been a time that chorus was printed.
                        # we should have loaded the word when first not in chorus into what_last_whale_said_array
                        # we do want to print if the last whale was in chorus before.
                        what_last_whale_said += " " + line['text']
                        what_last_whale_said_array.append(line['text'])

                    else: # not in a chorus, not previously in a chorus, and the whale is different.
                        if previous_whale_name and what_last_whale_said: # and, last whale said something, and not first entry
                            f.write(what_last_whale_said + ".\n")
                        what_last_whale_said = f"Whale {line['whale']}: {line['text']}"
                        what_last_whale_said_array = [line['text']]
                previously_in_chorus = False

            # we want to split up vocalizations that are a long time apart in text dialogue.
            if time_diff > max_diff and not np.isnan(time_diff) and not time_diff == np.inf:
                # this cannot be a chorus, as time_diff is high. So it would have printed.
                # Also print past vocalizations of the same whale (which otherwise would be skipped) because its a long pause.
                # we don't want to print this entry though, it needs to be printed after the pause (tdiff is prev - current time)
                if len(what_last_whale_said_array) > 1:
                    f.write(f"Whale {previous_whale_name}: " + " ".join(what_last_whale_said_array[:-1]) + ".\n")
                    # previous_whale_name = ""  # empty as there is no previous, it's the start of a conversation
                    what_last_whale_said = f"Whale {line['whale']}: {line['text']}"
                    what_last_whale_said_array = [line['text']]

                print_time_no_vocalizations(time_diff,f)
                time_diff = 0

            previous_whale_name = line["whale"]
            previous_whale_utterance = line["text"]
            last_timestamp = this_timestamp

        # After the loop, check if there's an unprocessed chorus
        if len(chorus_whales_data.keys()) > 0:
            print_chorus(chorus_whales_data, f)

        # Remember to write the last whale's sayings in each dialogue to the file
        f.write(what_last_whale_said + ".\n\n")

        # an extra newline before filenames to indicate significant separation
        f.write("\n")

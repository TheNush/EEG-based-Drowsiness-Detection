
#!/usr/bin/env python
import sys
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score
from hmm import HMM
from sklearn.ensemble import RandomForestClassifier
from file_handling import read_annotations_from_file
from file_handling import load_epochs_from_file
from feature_extraction import features_to_codebook
from feature_extraction import extract_features_from_epochs
import warnings



annotations_file = sys.argv[2]
sleep_stages_dict = {'Sleep stage W':5, 'Sleep stage 1':3, 'Sleep stage 2':2, 'Sleep stage 3':1,'Sleep stage 4':0, 'Sleep stage R':4, 'Movement time':6}
sleep_stages = read_annotations_from_file(annotations_file, sleep_stages_dict)
#print(sleep_stages)

# Main
# run: python src/dhmm.py data/SC4001E0-PSG.edf data/annotations.txt
# ====

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    """ Process the commandline arguments. Two arguments are expected: The .edf file path
    and the annotations .txt file path.
    """
    if len(sys.argv) == 3:
        edf_file = sys.argv[1]
        annotations_file = sys.argv[2]

        sleep_stages_dict = {'Sleep stage W':0, 'Sleep stage 1':1, 'Sleep stage 2':2, 'Sleep stage 3':3,
            'Sleep stage 4':4, 'Sleep stage R':5, 'Movement time':6}
        sleep_stages = read_annotations_from_file(annotations_file, sleep_stages_dict)
        sleep_stages = np.array(sleep_stages)
		#print(sleep_stages)
        nr_states = len(np.unique(sleep_stages))
        # annotations contain long sequences of the awake state at the beginning and the end - those are removed
        actual_sleep_epochs_indices = np.where(sleep_stages != sleep_stages_dict['Sleep stage W'])
        print(len(actual_sleep_epochs_indices[0]))
        #print(sleep_stages)
        sleep_start_index = actual_sleep_epochs_indices[0][0]
        sleep_end_index = actual_sleep_epochs_indices[0][-1]
        sleep_stages = sleep_stages[sleep_start_index:sleep_end_index]
        sleep_stages = np.array(sleep_stages)
        epochs = load_epochs_from_file(edf_file, epoch_length = 30, fs = 100)
        epochs = epochs[sleep_start_index:sleep_end_index,:]
        #print(epochs)
        #print(epochs.shape)

        features, mean = extract_features_from_epochs(epochs, epoch_length = 30, fs = 100)
       
        nr_groups = 20 # number of discrete features groups
        codebook, epoch_codes = features_to_codebook(features, nr_groups)
        print(epoch_codes.shape)
        print(epoch_codes)
        #print(features.shape)
        #print(mean)
        
        training_percentage = 0.8 # % of data used for training the model
        sleep_stages_train, sleep_stages_test = np.split(sleep_stages, [int(training_percentage * sleep_stages.shape[0])])         
        epoch_codes_train, epoch_codes_test = np.split(epoch_codes, [int(training_percentage * epoch_codes.shape[0])]) 
        
        #print(epoch_codes_train)
        #print("Now printing weird stuff...")
        #weird = Counter(zip(sleep_stages_train, epoch_codes_train)).items()
        #print(weird)
        hmm = HMM(nr_states, nr_groups)
        #print(sleep_stages_train)
        #print(epoch_codes_train)
        hmm.train(sleep_stages_train, epoch_codes_train)
        x=hmm.get_state_sequence(epoch_codes_test)
        '''

        sleep_stages_train = np.array(sleep_stages_train)
        epoch_codes_train = np.array(epoch_codes_train)
        sleep_stages_train = np.reshape(sleep_stages_train, ((115,1)))
        epoch_codes_train = np.reshape(epoch_codes_train, ((115,1)))

        print(sleep_stages_train.shape)
        print(epoch_codes_train.shape)
        print(epoch_codes_train)
        model = RandomForestClassifier()
        model.fit(sleep_stages_train, epoch_codes_train)
        x = model.predict(epoch_codes_test)
        '''


        sleep_stages_reverse = {y:x for x,y in sleep_stages_dict.items()}
        actual_phases = list(map(lambda phase: sleep_stages_reverse[phase], sleep_stages_test))
        predicted_phases = list(map(lambda phase: sleep_stages_reverse[phase], x))
        print("Actual sleep phases paired with predicted sleep phases:")
        f = open('C:/Users/Dell/Desktop/Projects/EEG Signal Processing/EEG_01/src/predicted.txt', 'w')
        for actual, predicted in zip(actual_phases, predicted_phases):
            print(actual, predicted)
            f.write(predicted)
            f.write('\n')
            
        print("Accuracy:", accuracy_score(sleep_stages_test, x))

    else:
        print('\n'.join(["Unexpected number of commandline arguments.",
                         "Usage: ./dhmm.py {edf-file} {annotation-file}"]))

  

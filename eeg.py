import  os
import numpy as np
from mne import Epochs, pick_types
from mne.io import find_edf_events, read_raw_edf
from collections import defaultdict
import matplotlib.pyplot as plt

class Eeg:

    gdf_files_names = []
    event_id = dict()
    data = dict()
    directory = ""

    def __init__(self, class1 = "left_hand", class2 = "right_hand"):
        print("Let's start with analyzing some signals!!\n")
        print("Firsly, you have to name classes by calling the method enter_names_of_classes()!")
        print("If you do not enter the names, they will be \"left_hand\" and \"right_hand\"!\n")

    @classmethod
    def get_gdf_name(cls, directory = "./"):
        cls.directory = directory
        for file in os.listdir(directory):
            if file.endswith(".gdf"):
                cls.gdf_files_names.append(file)

    @classmethod
    def enter_names_of_classes(cls, class1 = "left_hand", class2 = "right_hand"):
        cls.event_id[class1] = 769
        cls.event_id[class2] = 770

        cls.data[class1] = []
        cls.data[class2] = []

    @classmethod
    def get_raw_eeg_data(cls, tmin = 0., tmax = 4.):
        for file in cls.gdf_files_names:

            t_min, t_max = 0., 4.
            path = cls.directory + file

            raw_edf = read_raw_edf(path)
            events = cls.manage_edf_events(raw=raw_edf)
            picks = pick_types(raw_edf.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
            events = cls.exclude_rejected_trials(events=events)
            temp1 = Epochs(raw_edf, events, cls.event_id, t_min, t_max, proj=True, picks=picks, baseline=None,
                           preload=True)
            temp = temp1.drop_channels(ch_names={'EOG:ch01', 'EOG:ch02', 'EOG:ch03'})
            temp = temp.copy().crop(tmin=tmin.__float__(), tmax=tmax.__float__())

            keys = list(cls.event_id.keys())
            indices = list(range(len(keys)))
            for index, key in zip(indices,keys):
                cls.data[key].append(temp.__getitem__(keys[index]).get_data())

        cls.concatenate_data()

    @staticmethod
    def read_edf_file(input_name):
        raw = read_raw_edf(input_fname=input_name, preload=True, stim_channel='auto')
        #raw.rename_channels(lambda x: x.strip('.'))
        raw.filter(1., 50., fir_design='firwin', skip_by_annotation='edge')
        return raw

    @staticmethod
    def manage_edf_events(raw):
        # find_edf_events returns the list of five elements:
        # n_events - the number of all elements (this element
        #            excluded in order to transform list to array)
        # pos      - beginning of the events in samples
        # typ      - the event identifiers
        # chn      - the associated channels (0 for all)
        # dur      - duration of the events

        event = find_edf_events(raw)
        event1 = [event[1][:]]
        event2 = [event[2][:]]
        event3 = [event[3][:]]
        events = np.vstack((event1, event3, event2))
        events = events.transpose()
        return events

    @staticmethod
    def exclude_rejected_trials(events):

        # It has to be done because rejected trials crash Epochs function
        D = defaultdict(list)
        for i, item in enumerate(events[:, 0]):
            D[item].append(i)
        D = {k: v for k, v in D.items() if len(v) > 1}
        # indices of the rows to be deleted have to be appended
        # throughout one for loop and then performed deleting for all rows at once
        indices = []
        for key, value in D.items():
            indices.append(value)
        temp = np.delete(events, indices, 0)
        return temp

    @classmethod
    def concatenate_data(cls):
        keys = list(cls.data.keys())
        for key in keys:
            value = list(cls.data[key])
            cls.data[key] = np.concatenate(value, axis=0)

    def plot_signals(self, class_plt = "left_hand", sig_num = 0):
        subplots = [311,312,313]
        iter = [0,1,2]
        electrodes = ["C3", "Cz", "C4"]
        time_axis = np.arange(0,1001)/250
        plt.figure("Plot certain EEG signal over all three electrodes")
        for i, sp, electrode in zip(iter, subplots, electrodes):
            plt.subplot(sp)
            plt.plot(time_axis, self.data[class_plt][sig_num,i])
            plt.xlabel("Time (s)")
            plt.ylabel(("Electrode: "+electrode))
        plt.show()



eeg = Eeg()
eeg.get_gdf_name("../signals/")
eeg.enter_names_of_classes()
eeg.get_raw_eeg_data()
print(np.shape(eeg.data["left_hand"]))
print(np.shape(eeg.data["right_hand"]))
eeg.plot_signals()

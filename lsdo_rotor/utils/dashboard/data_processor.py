import os
import pickle
from datetime import datetime
from lsdo_rotor.utils.dashboard.dash_utils import DATETIME_FORMAT_STRING_FILENAME
import numpy as np


class DataProcessor():
    """
    The DataProcessor class is responsible for bookkeeping the data directory.
    It has the following functions:
    - Parse a data file given filepath
    - Output most recent data dictionary for plotting
    - Output full data dicionary for plotting
    - Check if a data file exists within a directory.
    """

    def __init__(self,
                 filetype='pkl',
                 data_dir_path='',
                 hist_var_names=[],
                 all_var_names={}):
        """
        DataProcessor class takes in the filetype and data directory path as arguments

        Parameters:
        ----------
            filetype: str
                string of type of data. Nominally set to '.pkl' (pickle file)
            data_dir_path: str
                string of absolute path to data directory. Nominally set to ''
            hist_var_names: list
                list of strings containg the variable names of all variables to keep full history of
        """

        self.filetype = filetype
        self.data_dir_path = data_dir_path
        self.hist_var_names = hist_var_names
        self.all_var_names = all_var_names

        # initialize data dict all
        self.data_dict_all = {}
        for c_ID in self.all_var_names:
            for var in self.all_var_names[c_ID]['hist_var_names']:

                if c_ID not in self.data_dict_all:
                    self.data_dict_all[c_ID] = {}
                self.data_dict_all[c_ID][var] = None
                self.data_dict_all[c_ID]['global_ind'] = []

        # initialize data dict current
        self.data_dict_current = {}
        for c_ID in self.all_var_names:
            for var in self.all_var_names[c_ID]['var_names']:
                self.data_dict_current[var] = {}
                self.data_dict_current[var]['global_ind'] = -1
                self.data_dict_current[var]['value'] = None
                self.data_dict_current[var]['local_ind'] = -1

        # Initialize indices dict
        self.indices_dict = {}
        self.indices_dict['global_ind'] = []
        for tuple in self.hist_var_names:
            client = tuple[0]
            self.indices_dict[client] = {}
            self.indices_dict[client]['local'] = []
            self.indices_dict[client]['global'] = []

        self.numpy_type = type(np.eye(3))
        self.sorted_files = []
        self.global_ind = []

    def add_data(self, filename, client_ID):
        """
        Parses and creates atrributes for data dictionary

        Parameters:
        ----------
            filename: str
                name of file to parse data from
        """
        # Data filepath
        data_file_filepath = self.abs_data_file_path(filename)

        # Open pickle file
        with open(data_file_filepath, 'rb') as handle:
            data_dict = pickle.load(handle)

        # Process indices.
        # UPDATE GLOBAL INDEX
        if len(self.indices_dict['global_ind']) == 0:
            self.indices_dict['global_ind'] = [0]
        else:
            self.indices_dict['global_ind'].append(
                self.indices_dict['global_ind'][-1] + 1)

        # UPDATE CLIENT INDEX
        if len(self.indices_dict[client_ID]['local']) == 0:
            self.indices_dict[client_ID]['local'] = [0]
            self.indices_dict[client_ID]['global'] = [
                self.indices_dict['global_ind'][-1]]
        else:
            self.indices_dict[client_ID]['local'].append(
                self.indices_dict[client_ID]['local'][-1] + 1)
            self.indices_dict[client_ID]['global'].append(
                self.indices_dict['global_ind'][-1])
        # ---- CURRENT DATA DICT ----
        # Fill out current data dict
        for var_name in data_dict:
            if var_name not in self.data_dict_current:
                continue
            self.data_dict_current[var_name]['value'] = data_dict[var_name]
            self.data_dict_current[var_name]['global_ind'] = self.indices_dict[client_ID]['global'][-1]
            self.data_dict_current[var_name]['local_ind'] = self.indices_dict[client_ID]['local'][-1]

        # ---- HISTORICAL DATA DICT ----
        self.add_all(data_dict, client_ID)

    def data_exists(self, filename):
        """
        Checks to see if a file exists within the data directory

        Parameters:
        ----------
            filepath: str
                Filepath of file being checked for existence

        Returns:
        ---------
            bool
                returns true if file exists and false otherwise
        """
        # Absolute path
        data_file_filepath = self.abs_data_file_path(filename)

        # Check if filename exists
        if not os.path.isfile(data_file_filepath):
            return False
        else:
            return True

    def abs_data_file_path(self, filename):
        """
        Gets the absolute file path using data directory filepath

        Parameters:
        ----------
            filename: str
                filename to return data file path

        Outputs:
        ----------
            str
                returns string of path to file
        """

        full_path = r'{0}/{1}'.format(self.data_dir_path, filename)
        return full_path

    def add_all(self, data_dict, client_ID):
        """
        Create data_dict_all

        Parameters:
        ----------
            data_dict: dictionary
                the data dictionary to add to full data dictionary
        """
        for tuple in self.hist_var_names:
            var_name = tuple[1]
            if var_name in data_dict:
                # IF data dict all entry is empty:
                if self.data_dict_all[client_ID][var_name] is None:
                    self.data_dict_all[client_ID][var_name] = [
                        data_dict[var_name]]
                # If it is not empty, go ahead and add data
                else:
                    self.data_dict_all[client_ID][var_name].append(
                        data_dict[var_name])

        self.data_dict_all[client_ID]['global_ind'] = self.indices_dict[client_ID]['global']

    def update_info(self):
        """
        searches through data directory and returns dict of information
        """
        indices_info_temp = {}
        indices_info_temp['global'] = []
        for tuple in self.hist_var_names:
            client = tuple[0]
            indices_info_temp[client] = []

        # List of all filenames within directory
        all_files = [f for f in os.listdir(self.data_dir_path) if (
            os.path.isfile(os.path.join(self.data_dir_path, f))) and f not in self.sorted_files]

        # print(self.data_dir_path, all_files)
        sorted_files = sorted(
            all_files,
            key=lambda x: datetime.strptime(
                x.rsplit('.')[2], DATETIME_FORMAT_STRING_FILENAME),
            reverse=False)
        sorted_clients = []

        # Go through all files and sort by date
        # Loop through all filenames
        for file in sorted_files:
            # Parse out client code
            client_code = file.rsplit('.')[1]
            sorted_clients.append(client_code)

            # Create dictionary entry for client code if not yet created
            if client_code not in indices_info_temp:
                indices_info_temp[client_code] = []

            # Add iteration number to respective iteration dictionary entry
            if len(indices_info_temp['global']) == 0:
                indices_info_temp['global'] = [0]
            else:
                indices_info_temp['global'].append(
                    indices_info_temp['global'][-1] + 1)

            # Add indices
            if len(indices_info_temp[client_code]) == 0:
                indices_info_temp[client_code] = [0]
            else:
                indices_info_temp[client_code].append(
                    indices_info_temp[client_code][-1] + 1)

            # Update attribute indices
            if len(self.global_ind) == 0:
                self.global_ind = [0]
            else:
                self.global_ind.append(self.global_ind[-1] + 1)

        # Return list
        return_dict = {}
        return_dict['global_ind'] = indices_info_temp
        return_dict['sorted_files'] = sorted_files
        return_dict['sorted_clients'] = sorted_clients

        if len(self.sorted_files) == 0:
            self.sorted_files = sorted_files
            self.sorted_clients = sorted_clients
        else:
            self.sorted_files = self.sorted_files + sorted_files
            self.sorted_clients = self.sorted_clients + sorted_clients

        return return_dict

    def update(self):
        """
        adds data that hasn't been processed yet to data dictionary
        """
        newest_info_dict = self.update_info()

        for iteration_num in newest_info_dict['global_ind']['global']:
            print('newest_info_dict:', iteration_num)

            # Get file name of pkl file containing the data
            data_file_name = newest_info_dict['sorted_files'][iteration_num]
            data_file_client = newest_info_dict['sorted_clients'][iteration_num]

            # If the next iteration data file does not exist, end loop
            if not self.data_exists(data_file_name):
                iterations_exist = False
                print('DATA FILE NOT FOUND: ', data_file_name)
                break

            # Read appropriate file from directory
            self.add_data(data_file_name, data_file_client)

    def get_current_data(self, ind):
        """
        return dictionary of data at index ind.
        """

        # Open pickle file
        # Absolute path
        data_file_filepath = self.abs_data_file_path(self.sorted_files[ind])
        with open(data_file_filepath, 'rb') as handle:
            data_dict = pickle.load(handle)

        current_client = self.sorted_clients[ind]
        current_dict = {}
        # Fill out current data dict
        for var_name in data_dict:
            current_dict[var_name] = {}
            current_dict[var_name]['value'] = data_dict[var_name]
            current_dict[var_name]['global_ind'] = self.indices_dict[current_client]['global'][ind]
            # print(ind, self.indices_dict[current_client]['global'][ind])

        # Return current dict
        return current_dict

    def get_plot_data(self, ind):
        """
        return full dictionary of data at index ind.
        """
        # CURRENT DATA:
        current_data = {}
        # print(self.global_ind)
        for client_ID in self.all_var_names:
            current_data[client_ID] = {}
            # Find the most recent value of global mapping for client:
            client_ind = self.get_client_global_ind(client_ID, ind)[0]

            # Open pickle file
            # Absolute path
            data_file_filepath = self.abs_data_file_path(
                self.sorted_files[client_ind])
            # print(self.indices_dict[client_ID]
            #       ['global'], client_ID, client_ind, ind)
            with open(data_file_filepath, 'rb') as handle:
                data_dict = pickle.load(handle)

            # Fill out current data dict
            for var_name in data_dict:
                current_data[client_ID][var_name] = data_dict[var_name]
                current_data[client_ID]['global_ind'] = [client_ind]

        # HIST DATA:
        historical_data = {}
        for client_ID in self.data_dict_all:

            # Find the most recent value of global mapping for client:
            list_ind = self.get_client_global_ind(client_ID, ind)[1]

            historical_data[client_ID] = {}
            historical_data[client_ID]['global_ind'] = self.indices_dict[client_ID]['global'][:list_ind+1]
            for var_name in self.data_dict_all[client_ID]:
                # print(var_name)
                historical_data[client_ID][var_name] = np.array(self.data_dict_all[client_ID][var_name])[0:list_ind]

        return historical_data, current_data

    def get_client_global_ind(self, client_ID, ind):
        """
        returns global client index less than ind
        """
        # Find the most recent value of global mapping
        c_ind_previous = self.indices_dict[client_ID]['global'][0]
        for list_index, c_ind in enumerate(self.indices_dict[client_ID]['global']):
            if c_ind > ind:
                # client ind is the greatest global index in the client data.
                client_ind = c_ind_previous
                list_ind = list_index
                break
            c_ind_previous = c_ind

            # If very last data set:
            if c_ind == self.indices_dict[client_ID]['global'][-1]:
                client_ind = c_ind_previous
                list_ind = list_index+1

        # return value
        return client_ind, list_ind

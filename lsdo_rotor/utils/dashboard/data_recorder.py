from lsdo_rotor.utils.dashboard.dash_utils import DATETIME_FORMAT_STRING_FILENAME, create_file_name

from datetime import datetime
import pickle


class DataRecorder():
    """
    The DataProcessor class is responsible for bookkeeping the data directory.
    It has the following functions:
    - Parse a data file given filepath
    - Output most recent data dictionary for plotting
    - Output full data dicionary for plotting
    - Check if a data file exists within a directory.
    """

    def __init__(self,
                 dash_instance):

        self.dash_instance = dash_instance

        self.local_iteration = {}
        for client in self.dash_instance.vars:
            self.local_iteration[client] = 0

        self.var_collection_dict = self.dash_instance.vars

    def record(self, data_dictionary, client_ID):
        """
        Called by clients to save data to user defined directory.
        'Clients' refer to the object saving data. For example, ozone or csdl_om.simulator
        Clients simply call this method whenever a model evaluation is finished by entering a dictionary of data and their respective client IDE.

        client        record method         data directory
        ------        -------------         --------------
        all data -->    parse data   -->    save as pkl to data dir

        Saved filename format:
        'user-prefix'.'client'.'timestamp'.'extension'

        User-Prefix:    Defined by user in the configuration setup method
        Client:         Client ID given as an argument to this method
        Timestamp:      Date/time at the time the data was received. Done within this method
        Extension:      File format to save. Nominally '.pkl'

        An example of data saved from the ODEproblem class from the ozone package:
        data_entry.ozone.2021-10-10-12_12_12.pkl

        Parameters:
        ----------
            data_dictionary: Dict
                A dictionary where the key is the name of the variable and the value is the variable itself.
                A client passes through an entire dictionary of data and this method parses out the necessary variables.
            client_ID: str
                A unique code associated with the client
        """

        # Check whether we save or not
        self.local_iteration[client_ID] += 1
        stride = self.dash_instance.vars[client_ID]['stride']

        if stride == 1:
            pass
        elif int((self.local_iteration[client_ID]-1) % stride) != 0:
            return

        # Current timestamp string
        timestamp = datetime.now().strftime(DATETIME_FORMAT_STRING_FILENAME)

        # filename string
        filename = create_file_name(
            self.dash_instance.data_file_name, client_ID, timestamp, self.dash_instance.data_file_type)

        # With the data to save and its filename, save to relevant directory.
        if self.dash_instance.data_file_type == 'pkl':
            # Get filpath to save
            data_path = r'{0}/{1}'.format(
                self.dash_instance.data_dir_path, filename)
            print(data_path)
            # Save as pickle file
            with open(data_path, 'wb') as handle:
                pickle.dump(data_dictionary, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

    def record_external(self, file_name, object_to_save):
        """
        Saves non-solver object_to_save as pkl file to external_dir_path. or try at least.
        """

        # filename string
        filename = create_file_name(file_name, self.dash_instance.data_file_type)

        # With the data to save and its filename, save to relevant directory.
        if self.dash_instance.data_file_type == 'pkl':
            # Get filpath to save
            data_path = r'{0}/{1}'.format(
                self.dash_instance.external_dir_path, filename)
            print(data_path)
            # Save as pickle file
            with open(data_path, 'wb') as handle:
                pickle.dump(object_to_save, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

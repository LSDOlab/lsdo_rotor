from lsdo_dash.utils import get_abs_path, process_date_time, create_file_name, dir_exists, date2string
from lsdo_dash.data_processor import DataProcessor
from lsdo_dash.frame import Frame
from cv2 import VideoWriter_fourcc, VideoWriter, imread
import os
from lsdo_dash.data_recorder import DataRecorder
from lsdo_dash.interface import Interface

from datetime import datetime
import pickle


class BaseDash():
    """
    BaseDash has the responsibility of creating the frames and movie of the optimization problem.
    The default configuration values are as follows:

    """

    def __init__(self):
        # Default configuration values:
        # File names
        self.run_file_name = 'example_script.py'
        self.data_file_name = 'data_entry'

        # Directories
        self.frames_dir = '_frames'
        self.movies_dir = '_movies'
        self.data_dir = '_data'
        self.external_dir = '_external'

        # Visualization settings
        self.fps = 30
        self.stride = 1
        self.write_stride = 1
        self.savefig_dpi = 250
        self.print_opt_variables = True
        self.print_nonlinear_status = False
        self.print_linear_status = False
        self.frame_name_prefix = 'output'
        self.movcodec = 'DIVX'
        self.movtype = 'mp4'

        # read user defined configuration variables
        self.set_config()
        self.client_setup_current = None

        # Case archive directory path
        self.case_archive_path = os.getcwd() + '/' + '_case_archive'
        self.timestamp_path = os.getcwd() + '/' + 'null_timestamp'
        self.timestamp_name = 'null_timestamp'
        self.recording_started = False

        if not dir_exists(self.case_archive_path):
            # If case archive directory does not exist, create directory
            os.mkdir(self.case_archive_path)
            print('ARCHIVE DIRECTORY CREATED IN ', self.case_archive_path)
        else:
            # If it does exist, use most recent timestamp as current timestamp as long as there are folders in _case_archive
            case_archive_is_empty = True
            for check_ca_file in os.listdir(self.case_archive_path):
                if os.path.isdir(self.case_archive_path + '/'+check_ca_file):
                    case_archive_is_empty = False

            if case_archive_is_empty:
                print('NO DATA FOUND IN ', self.case_archive_path)
            else:
                self.use_timestamp(initial=True)

        # Define data directory path/frames directory path
        self.data_dir_path = get_abs_path(
            self.timestamp_path,
            self.data_dir)
        self.frame_dir_path = get_abs_path(
            self.timestamp_path, self.frames_dir)
        self.movie_dir_path = get_abs_path(
            self.timestamp_path, self.movies_dir)
        self.external_dir_path = get_abs_path(
            self.timestamp_path, self.external_dir)

        # initialize list of variable names to store
        self.hist_var_names = []
        self.vars = {}
        # DEFAULT VARIABLES:
        # self.save_variable('objective', 'optimizer')
        # self.save_variable('constraint1', 'optimizer')
        # self.save_variable('constraint2', 'optimizer')

        # Dictionary of frames
        self.frame = {}
        self.frame_GUI = {}

        # Run the setup method to figure out what values are needed:
        self.setup()

        # TEMPORARY:
        self.data_file_type = 'pkl'

        # Create a dataprocessor object
        self.create_dataprocessor()

    def use_timestamp(self, date='', time='', initial=False):
        """
        Takes a user-specified date/time and returns the name of the folder ('YYYY-MM-DD-HH_MM_SS/')
        in the case archive directory closest to given timestamp.

        Parameters:
            date: str
                string of date with format "YYYY-MM-DD".If left empty, current date is used
            time: str
                string of time with format "HH_MM_SS".If left empty, current time is used
        """
        # the absolute path of the current time step
        timestep_name_temp = process_date_time(
            self.case_archive_path, date=date, time=time)

        if self.timestamp_name == timestep_name_temp:
            return
        else:
            self.timestamp_name = timestep_name_temp

        self.timestamp_path = self.case_archive_path + '/' + self.timestamp_name

        # Define data directory path/frames directory path
        self.data_dir_path = get_abs_path(
            self.timestamp_path,
            self.data_dir)
        self.frame_dir_path = get_abs_path(
            self.timestamp_path, self.frames_dir)
        self.movie_dir_path = get_abs_path(
            self.timestamp_path, self.movies_dir)
        self.external_dir_path = get_abs_path(
            self.timestamp_path, self.external_dir)

        # Create dataprocessor object for new timestamp
        if not initial:
            # Create a dataprocessor object
            self.create_dataprocessor()

    def create_dataprocessor(self):
        """
        set data processor instance attribute as new instance for this timestamp
        """
        self.data_processor = DataProcessor(
            filetype=self.data_file_type,
            data_dir_path=self.data_dir_path,
            hist_var_names=self.hist_var_names,
            all_var_names=self.vars)

    def set_config(self):
        """
        Gets called to set configuration variables. Defined by user.
        The user must define the following optional variables :
            self.run_file_name: string of optimization script name
            self.data_dir: string of name of data directory
            self.data_file_name: string of prefix of data pkl files within data directory
            self.frames_dir: string of name of frames directory
            self.fps:
            self.stride:
            self.write_stride:
            self.savefig_dpi:
            self.print_opt_variables:
            self.print_nonlinear_status:
            self.print_linear_status:
            self.frame_name_prefix = 'output_{}'
            self.movcodec = 'DIVX'
            self.movtype = 'avi'

        If any of the previous variables were not defined, they are defaulted to:
            self.run_file_name = 'run.py'
            self.data_dir = '_data'
            self.data_file_name = 'opt'
            self.frames_dir = '_frames'
            self.fps = 30
            self.stride = 1
            self.write_stride = 1
            self.savefig_dpi = 250
            self.print_opt_variables = True
            self.print_nonlinear_status = False
            self.print_linear_status = False
            self.frame_name_prefix = 'output'
            self.movcodec = 'DIVX'
            self.movtype = 'avi'

        """
        pass

    def visualize_auto_refresh(self, refresh_time=1.0):
        """
        displays a pyplot instance that updates with new data every <refresh_time> seconds.
        """

        import matplotlib.pyplot as plt

        print('Kill process to stop auto-refresh')
        while True:
            print('refreshing...')
            self.visualize_most_recent(show=False)
            plt.draw()
            plt.pause(refresh_time)

    def reset_saved_frames(self):
        """
        deletes all images in the _frames directory.
        """
        dir = self.frame_dir_path
        for f in os.listdir(dir):
            filename_splits = os.path.splitext(f)
            if filename_splits[-1] == '.png':
                print(f'REMOVING FILE: {f}')
                os.remove(os.path.join(dir, f))

    def visualize_all(self, date='', time='',  **kwargs):
        """
        Runs the procedure to create all frames for all iterations given a dataset date/time string.
        Note that an optimization must be completed with all data written to the specified data directory.
        If a date and time are not specified, the most recent data is used. Calls the method visualize().

        Parameters:
        ----------
            date: str
                date of the directory within the data directory containing specified data. If both date and time are emptry/not specified, the mosr recent data will be used.

            time: str
                time of the directory within the data directory containing specified data. If both date and time are emptry/not specified, the mosr recent data will be used.

        """
        # Call visualize with initial and final == false
        self.visualize(**kwargs)

    def visualize_first(self, frame_ind='all',  **kwargs):
        """
        This method runs the procedure to create the first frame for all iterations given a dataset date/time string.
        Note that an optimization must be running with data being written to the specified data directory.
        If a date and time are not specified, the most recent data is used. Calls the method visualize().

        Parameters:
        ----------
            date: str
                date of the directory within the data directory containing specified data. If both date and time are emptry/not specified, the mosr recent data will be used.

            time: str
                time of the directory within the data directory containing specified data. If both date and time are emptry/not specified, the mosr recent data will be used.

        """
        # Call visualize with initial and final == false
        self.visualize(frame_ind=0,  **kwargs)

    def visualize_most_recent(self,  **kwargs):
        """
        This method runs the procedure to create the most recent frame for all iterations given a dataset date/time string.
        Note that an optimization must be running with data being written to the specified data directory.
        If a date and time are not specified, the most recent data is used. Calls the method visualize().

        Parameters:
        ----------
            date: str
                date of the directory within the data directory containing specified data. If both date and time are emptry/not specified, the mosr recent data will be used.

            time: str
                time of the directory within the data directory containing specified data. If both date and time are emptry/not specified, the mosr recent data will be used.
        """
        # Call visualize with initial and final == false
        self.visualize(frame_ind='most recent',  **kwargs)

    def update_data_processor(self):
        """
        update data processor to read in all data.
        """
        self.data_processor.update()

    def visualize(self, frame_ind='all', location_type='standard', show=False):
        """
        This method runs the procedure to create all frames for all iterations. Containts a loop for all iterations that calls create_frame
        """

        # Update data processor
        self.update_data_processor()

        # Update all frame objects with correct date_time_names
        for frame in self.frame.values():
            frame.date_time_directory = self.data_dir_path

        # Check if _frames directory exists. if not, create _frames
        if not os.path.isdir(self.frame_dir_path):
            os.mkdir(self.frame_dir_path)

        # Get list of iteration numbers saved in the data directory.
        self.data_processor.update_info()
        data_indices = self.data_processor.global_ind
        # Create list of iteration numbers to visualize for
        if frame_ind == 'all':
            # iteration list = all saved iteration data
            iteration_list = data_indices
        if frame_ind == 'most recent':
            # If most recent frame is requested, use last iteration
            iteration_list = [data_indices[-1]]
        if type(frame_ind) == type(1):
            # If frame_ind is an integer, get that index
            iteration_list = [frame_ind]
        elif type(frame_ind) == type([1, 1]):
            # If frame_ind is a list, iteration list = frame_ind
            iteration_list = frame_ind

        # Visualization loop
        for iteration_num in iteration_list:
            print('VISUALIZING ITERATION:', iteration_num)

            # For each frame, update the iteration number
            for frame in self.frame.values():
                frame.iteration_num = iteration_num
                if show == True:
                    frame.show = True
                else:
                    frame.show = False
            self.create_frame(iteration_num, location_type)

        if show == True:
            import matplotlib.pyplot as plt
            plt.show()

    def create_frame(self, ind, location_type):
        """
        Processes raw data from data directory to feed into user defined plot method and calls the plot method for an iteration.
        """

        limits_dict = [1, 1]
        # Create inputs for plot method:
        # - Current iteration dictionary
        # - All iterations dictionary
        # - Limits
        (historical_data, current_data) = self.data_processor.get_plot_data(ind)
        if location_type == 'standard':
            self.plot(self.frame,
                      current_data,
                      historical_data,
                      limits_dict,
                      video=False)
        elif location_type == 'GUI':
            self.plot(self.frame_GUI,
                      current_data,
                      historical_data,
                      limits_dict,
                      video=False)
            # Call the plot method

    def plot(self,
             data_dict_current,
             data_dict_all,
             limits_dict,
             video=False):
        """
        Gets called to create a frame given optimization output. Defined by user.
        The method has the following high level structure:
        - Read variables from argument
        - call "clear_frame" method for each frame
        - plot variables onto each frame
        - call "save_frame" method for each frame

        Parameters
        ----------
            data_dict_current: dictionary
                dictionary where keys (names of variable) contain current iteration values of respective variable
            data_dict_all: dictionary
                dictionary where keys (names of variable) contain all iteration values of respective variable
            limits_dict: dictionary
                dictionary where keys (names of variables) contain [min, max] of respective variable
            ind: int
                integer of current optimization iteration
            video: bool
                what is this again
        """
        pass

    def setup(self):
        """
        Gets called to prepare what frames will be created. Defined by user.
        """
        pass

    def save_variable(self, variable_name, history=True):
        """
        Creates frames (images) using user-defined plot method for each optimization iteration and saves to a frames directory.
        """

        if not self.client_setup_current:
            raise ValueError('current client ID not found. call `self.set_clientID(<client>)` before `self.save_variable`. (<client> = `simulator` for CSDL)')

        self.vars[self.client_setup_current]['var_names'].append(variable_name)
        if history == True:
            self.vars[self.client_setup_current]['hist_var_names'].append(variable_name)
            self.hist_var_names.append((self.client_setup_current, variable_name))

    def save_python_object(self, file_name, python_object):
        """
        saves an instance of a non-solver variable for availability during user plotting functions.
        access this object's data in def plot by using self.get_external_data(file_name)

        Parameters:
        -----------
            python_object: 
                an object that can be saved and loaded as a pickle file.
        """
        # If recording started for this instance, do not create new directory with times.
        if self.recording_started == False:
            # make a new folder for the data from this specific runtime timestamp
            datetime_string = date2string(datetime.now())
            self.create_new_timestamp_entry(datetime_string)

            # Set recording_started to True so the instance does not create more timestamp entries
            self.recording_started = True

            self.data_recorder = DataRecorder(self)

        # save using recorder
        self.data_recorder.record_external(file_name, python_object)

    def get_external_data(self, file_name):
        """
        return the python object saved by user in save_python_object.

        Parameters:
        -----------
            file_name: 
                name of file saved using save_python_object
        """
        filename = create_file_name(file_name, self.data_file_type)

        # With the data to save and its filename, save to relevant directory.
        if self.data_file_type == 'pkl':
            # Get filpath to save
            data_path = r'{0}/{1}'.format(
                self.external_dir_path, filename)

        with open(data_path, 'rb') as handle:
            data_dict = pickle.load(handle)

        return data_dict

    def set_clientID(self, client_ID, write_stride=1):
        """
        method used in setup to declare which variables to save. Call this method to set up a client, and call save_variable afterwards.

        Currently available clients:
        - <CLIENT ID>  : object
        - 'simulator'  : csdl_om.Simulator or python_csdl_backend.Simualator
        
 
        """
        self.client_setup_current = client_ID
        # Initialize dictionary
        if client_ID not in self.vars:
            self.vars[client_ID] = {}
            self.vars[client_ID]['var_names'] = []
            self.vars[client_ID]['hist_var_names'] = []
            self.vars[client_ID]['stride'] = write_stride

    def add_frame(self,
                  name,
                  width_in=5.,
                  height_in=5.,
                  nrows=1,
                  ncols=1,
                  wspace=0.2,
                  hspace=0.2,
                  keys3d=[],
                  **kwargs,):

        # Create standard frame for saving image
        self.frame[name] = Frame(name,
                                 width_in=width_in,
                                 height_in=height_in,
                                 nrows=nrows,
                                 ncols=ncols,
                                 wspace=wspace,
                                 hspace=hspace,
                                 keys3d=keys3d,
                                 **kwargs,)

        self.frame[name].frame_dir_path = self.frame_dir_path
        self.frame[name].savefig_dpi = self.savefig_dpi
        self.frame[name].frame_name_prefix = self.frame_name_prefix

        # Create GUI frame for GUI
        self.frame_GUI[name] = Frame(name,
                                     type='GUI',
                                     width_in=width_in,
                                     height_in=height_in,
                                     nrows=nrows,
                                     ncols=ncols,
                                     wspace=wspace,
                                     hspace=hspace,
                                     keys3d=keys3d,
                                     **kwargs,)

        self.frame_GUI[name].frame_dir_path = self.frame_dir_path
        self.frame_GUI[name].savefig_dpi = self.savefig_dpi
        self.frame_GUI[name].frame_name_prefix = self.frame_name_prefix

    def make_mov(self):
        """
        This method creates a movie in the .avi file format, based on .png images from running the optimizer to
        generate frames. Movie parameters are set in the user created set_config method of the user facing dash
        class.
        """
        # First check if a frame directory exists. If not, run_visualization
        if not os.path.isdir(self.frame_dir_path):
            self.visualize_all()

        # set pathin and pathout based on current directory
        pathIn = self.frame_dir_path

        # Check if movie date/time directory exists. if not, create date/time directory
        if not os.path.isdir(self.movie_dir_path):
            os.mkdir(movie_dir_path_dt)

        # Try to read in png files from frames directory and again
        all_files = [f for f in os.listdir(pathIn) if (
            os.path.isfile(os.path.join(pathIn, f)) and f.endswith('png'))]

        # If there are no images in this directory, run_visualization for this date and time
        if len(all_files) == 0:
            print_str = r'No image files found in {0}. Creating frames'.format(
                self.frame_dir_path)
            print(print_str)
            self.visualize_all()

            # Try to read in png files from frames directory and again
            all_files = [f for f in os.listdir(pathIn) if (
                os.path.isfile(os.path.join(pathIn, f)) and f.endswith('png'))]

        # look within files to parse out frame names
        # initialize dict of frame_names
        frame_name_dict = {}
        for file in all_files:
            # parse out frame_name
            temp_frame_format = file.rsplit('.')[1]

            # Create frame_name dictionary entry if frame_name is unique
            if temp_frame_format not in frame_name_dict:
                frame_name_dict[temp_frame_format] = []

            # Add file of frame_name to appropriate dictionary entry
            frame_name_dict[temp_frame_format].append(file)

        # Create a separate movie for each frame_name
        for frame_name in frame_name_dict:
            frame_array = []
            file_temp = frame_name_dict[frame_name]
            # print(file_temp[1])
            files = sorted(file_temp, key=lambda f: int(f.split('.')[2]))
            print(files)

            # Create path to save movie
            pathOut = self.movie_dir_path + '/movie_' + frame_name + '.' + self.movtype

            # for each file, get the full filepath and properties of each image, adding to frame array
            for i in range(len(files)):
                filename = pathIn + '/'+files[i]
                print('filename:', filename)
                # reading each files
                img = imread(filename)
                height, width, layers = img.shape
                size = (width, height)

                # inserting the frames into an image array
                frame_array.append(img)

            # using frames, use a fourcc codec standard to encode the video
            # more info on DIVX and other codecs: https://www.fourcc.org/divx/
            out = VideoWriter(
                pathOut, VideoWriter_fourcc(*self.movcodec), self.fps, size)

            for i in range(len(frame_array)):
                # writing to a image array using python opencv
                out.write(frame_array[i])

            # Create file
            out.release()

    def run_GUI(self, plot_default=True, plot_user=True):
        """
        This method creates and runs the browser output
        """

        # Interface object deals with TK GUI
        interface = Interface(
            self, plot_default=plot_default, plot_user=plot_user)
        interface.run_GUI()

    def get_recorder(self):
        """
        This method makes a directory for data storage, and allows the clients,
        such as the simulator and optimizer, to access the writer method. The client (csdl_om.Simulator, ozone, etc)
        stores this method and passes a data dictionary through them. Specifics included in DataRecorder.record
        """
        # If recording started for this instance, do not create new directory with times.
        if self.recording_started == False:
            # make a new folder for the data from this specific runtime timestamp
            datetime_string = date2string(datetime.now())
            self.create_new_timestamp_entry(datetime_string)

            # Set recording_started to True so the instance does not create more timestamp entries
            self.recording_started = True

            self.data_recorder = DataRecorder(self)

        # return the writer method from the DataRecorder instance
        return self.data_recorder

    def create_new_timestamp_entry(self, datetime_string):
        """
        creates the relevant directories for new timestamp
        """

        print('NEW DATA RECORDING SESSION STARTED AT TIMESTAMP: ', datetime_string)
        # New timestep path
        self.timestamp_path = self.case_archive_path + '/' + datetime_string
        os.mkdir(self.timestamp_path)

        # Define data directory path/frames directory path and create directories
        self.data_dir_path = get_abs_path(
            self.timestamp_path,
            self.data_dir)
        self.frame_dir_path = get_abs_path(
            self.timestamp_path,
            self.frames_dir)
        self.movie_dir_path = get_abs_path(
            self.timestamp_path,
            self.movies_dir)
        self.external_dir_path = get_abs_path(
            self.timestamp_path,
            self.external_dir)

        os.mkdir(self.data_dir_path)
        os.mkdir(self.frame_dir_path)
        os.mkdir(self.movie_dir_path)
        os.mkdir(self.external_dir_path)

        self.use_timestamp()

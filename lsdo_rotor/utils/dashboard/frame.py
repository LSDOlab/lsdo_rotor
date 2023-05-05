import matplotlib.pyplot as plt
import matplotlib
from lsdo_rotor.utils.dashboard.dash_utils import create_file_name, WIDTH_GUI_PLOT, HEIGHT_GUI_PLOT
# matplotlib.use('Agg')
# matplotlib.use("TkAgg")


class Frame():
    """
    The frame class has the responsiblity of transferring information from
    plotting instructions given by the user in "Dash.plot()" to matplotlib figure instructions.
    It has the following tasks:
    - Create a figure
    - Save figure as image file to a directory
    - allow user to plot to figure
    """

    def __init__(self,
                 name,
                 type='save',
                 width_in=5.,
                 height_in=5.,
                 nrows=1,
                 ncols=1,
                 wspace=0.2,
                 hspace=0.2,
                 keys3d=[],
                 **kwargs,):

        self.frame_name = name
        self.ext_name = 'png'
        self.frame_type = type
        self.show = False

        self.width_in = width_in
        self.height_in = height_in
        if self.frame_type == 'save':
            self.fig = plt.figure(figsize=(width_in, height_in))
            # plt.close(self.fig)
        elif self.frame_type == 'GUI':
            # self.fig = plt.figure(figsize=(WIDTH_GUI_PLOT, HEIGHT_GUI_PLOT))
            self.fig = plt.figure(figsize=(width_in, height_in))
            plt.close(self.fig)

        self.gs = self.fig.add_gridspec(
            nrows=nrows,
            ncols=ncols,
            wspace=wspace,
            hspace=hspace,
        )

        self.axes = dict()

        self.keys3d = keys3d

        self.frame_dir_path = ''
        self.date_time_directory = ''
        self.savefig_dpi = 0
        self.iteration_num = 0
        self.frame_name_prefix = ''

    def __getitem__(self, gs_key):
        """
        Returns a 'matplotlib.axes._subplots.AxesSubplot' object to perform plotting operations on to the self.fig attribute

        Parameters:
        ----------

        Outputs:
        ----------
            'matplotlib.axes._subplots.AxesSubplot'
        """

        # Read subplot coordinates
        key0 = gs_key[0]
        key1 = gs_key[1]

        # Check if multiple subplots are being called
        if isinstance(key0, slice):
            key0 = (key0.start, key0.stop, key0.step)
        if isinstance(key1, slice):
            key1 = (key1.start, key1.stop, key1.step)

        key = (key0, key1)

        # When we plot the first time, create subplot
        if key not in self.axes:
            if key in self.keys3d:
                self.axes[key] = self.fig.add_subplot(
                    self.gs[gs_key],
                    projection='3d',
                )
            else:
                self.axes[key] = self.fig.add_subplot(self.gs[gs_key])
        return self.axes[key]

    def write(self):
        """
        Saves the current figure to the frames directory with specified resolution.
        The name of the saved file format is:
        """

        if self.frame_type == 'save':
            # Create string of path
            frame_path = self.get_frame_path(
                self.frame_dir_path,
                self.frame_name_prefix,
                self.frame_name,
                self.iteration_num,
                self.ext_name)

            # Temporarily storing full frame_path
            self.full_frame_path = frame_path

            # Saves the figure with given resolution
            self.fig.savefig(frame_path, dpi=self.savefig_dpi)
            if self.show == True:
                self.fig.show()

                # if self.frame_type  == 'save':
                #     plt.show()

                # dummy = self.fig
                # new_manager = dummy.canvas.manager
                # new_manager.canvas.figure = self.fig
                # self.fig.set_canvas(new_manager.canvas)

                # dummy.show()
                # plt.show()

    def clear_all_axes(self):
        """
        Clears the data from all axes
        """
        for key in self.axes:
            self.axes[key].clear()

    def get_frame_path(self, frame_dir, frame_prefix, frame_name, iteration_num, ext):
        """
        This method finds the path to save a frame given the frame directory, date, frame name and iteration number
        """
        # File name:
        file_name = create_file_name(
            frame_prefix, frame_name, iteration_num, ext)

        # return path:
        return r'{0}/{1}'.format(frame_dir,  file_name)

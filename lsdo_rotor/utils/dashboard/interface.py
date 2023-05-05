import tkinter as tk
from tkinter import ttk
import os
from lsdo_rotor.utils.dashboard.dash_utils import WIDTH_GUI_PLOT, HEIGHT_GUI_PLOT, WIDTH_INTERFACE, HEIGHT_INTERFACE, string2string

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")


class Interface:

    def __init__(self, dash, plot_default=True, plot_user=True):

        # Dashboard instance attribute
        self.dash = dash
        self.DEFAULT = plot_default
        self.USER = plot_user

        # Create main window regardless of what's being plotted
        self.root = tk.Tk()
        self.root.wm_title("LSDO_Dash")
        self.root.geometry(r'{0}x{1}'.format(
            WIDTH_INTERFACE, HEIGHT_INTERFACE))
        self.root.minsize(int(WIDTH_INTERFACE/1.5), int(HEIGHT_INTERFACE/1.5))
        # Make window resize properly
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Other TK variables
        self.auto_ref = tk.BooleanVar(self.root)
        self.current_timestamp = tk.StringVar(self.root)
        self.current_timestamp.trace('w', self.change_timestamp)
        self.functionboxvals = tk.Variable(self.root, value=[])
        self.timestamp_list = []
        self.slider_iteration = tk.IntVar(self.root)

        s = ttk.Style(self.root)
        s.theme_use('classic')
        s.map('TNotebook.Tab', background=[
              ('selected', 'white')], foreground=[('active', 'grey')])
        s.map('TNotebook')
        # s.configure('TFrame', background="white")

        self.notebook_frame = ttk.Frame(
            self.root, borderwidth=1, relief='solid')
        self.notebook_frame.pack(
            side=tk.RIGHT, pady=10, padx=(0, 10), expand=True, fill=tk.BOTH)

        self.notebook = ttk.Notebook(self.notebook_frame, style='TNotebook')
        self.notebook.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # ------- SHARED OPTIONS FRAME -------:
        self.sharedframe = ttk.Frame(
            self.root, borderwidth=1, relief='solid')
        self.sharedframe.pack(side=tk.LEFT, fill=tk.BOTH,
                              expand=0, padx=10, pady=10)
        self.sharedframe.columnconfigure(0, weight=1)
        self.sharedframe.rowconfigure(0, weight=1)

        # Status text vars
        #self.string1 = ''
        #self.string2 = ''
        self.status_list = []

        if self.DEFAULT:
            # ------- TAB 1 -------: DEFAULT PLOT
            # VARIABLES:
            # - frame_tab1:         frame for tab 1
            # - graphframe_tab1:    frame containing plot in frame_tab1
            # - figure_tab1:        MPL figure for tab 1
            # - canvas_tab1:        figure_tab1 canvas_tab1

            # Add main frame. All widgets are added to main frame
            self.frame_tab1 = ttk.Frame(self.notebook)
            self.frame_tab1.grid()

            # Plotting for TAB 1=
            self.graphframe_tab1 = ttk.Frame(self.frame_tab1)

            # Instantiate the MPL figure
            self.figure_tab1 = plt.figure(
                figsize=(WIDTH_GUI_PLOT, HEIGHT_GUI_PLOT), dpi=100, facecolor="white")
            plt.plot([0.0, 0.01, 0.02], [0.0, 0.01, 0.02])

            # Link the MPL figure onto the TK canvas_tab1 and pack it
            self.canvas_tab1 = FigureCanvasTkAgg(
                self.figure_tab1, master=self.graphframe_tab1)
            self.canvas_tab1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            # self.canvas_tab1.get_tk_widget().grid(row=0, column=0)
            self.graphframe_tab1.grid(row=0, column=0)

            # Add a toolbar to explore the figure like normal MPL behavior
            toolbar = NavigationToolbar2Tk(
                self.canvas_tab1, self.graphframe_tab1)
            toolbar.update()
            self.canvas_tab1._tkcanvas.pack(
                side=tk.TOP, fill=tk.BOTH, expand=1)

            # ------ NOTEBOOK -------
            # Add tabs for the two frames
            self.notebook.add(self.frame_tab1, text='DEFAULT PLOTS')

        if self.USER:
            # ------- TAB 2 -------:
            # - frame_tab2:         frame for tab 2
            # - graphframe_tab2:    frame containing plot in frame_tab2
            # - figure_tab2:        MPL figure for tab 2
            # - canvas_tab2:        figure_tab1 canvas_tab1
            # CREATING SECOND FRAME:

            self.frame_tab2 = ttk.Frame(self.notebook)
            # self.frame_tab2.grid()
            self.frame_tab2.pack()

            # Plotting for TAB 2
            self.graphframe_tab2 = ttk.Frame(self.frame_tab2)

            # Instantiate the MPL figure
            # self.figure_tab2 = plt.figure(
            #     figsize=(4.0, 4.0), dpi=100, facecolor="white")
            # plt.plot([0.0, 0.01, 0.02], [0.0, -0.01, -0.02])

            # Tab 2 plots the first frame defined by user.
            for frame_name in self.dash.frame:
                self.figure_tab2 = self.dash.frame_GUI[frame_name].fig
                break

            # Link the MPL figure onto the TK canvas_tab1 and pack it
            self.canvas_tab2 = FigureCanvasTkAgg(
                self.figure_tab2, master=self.graphframe_tab2)
            self.canvas_tab2.get_tk_widget().pack(side="top", fill='both', expand=True)
            # self.canvas_tab2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            # self.canvas_tab1.get_tk_widget().grid(row=0, column=0)
            # self.graphframe_tab2.grid(row=0, column=0)
            self.graphframe_tab2.pack(side="top", fill='both', expand=True)

            # Add a toolbar to explore the figure like normal MPL behavior
            toolbar = NavigationToolbar2Tk(
                self.canvas_tab2, self.graphframe_tab2)
            toolbar.update()
            self.canvas_tab2._tkcanvas.pack(
                side="top", fill='both', expand=True)
            # self.canvas_tab2._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            # ------ NOTEBOOK -------
            # Add tabs for the two frames
            self.notebook.add(self.frame_tab2, text='USER PLOTS')

    def quit(self):
        """
        Destroy GUI window cleanly if quit button pressed.
        """
        self.root.quit()
        self.root.destroy()

    def refresh_data_button(self):
        self.update_status("Refreshing Data...")
        self.refresh_data()

    def refresh_data(self):
        # Reads data from data directory through basedash class
        # self.dash.refresh()
        print(self.legendval.get())
        # Update both plots
        if self.USER:
            # update user defined plots
            self.update_user_plots()
            self.iteration_slider.configure(
                to=self.dash.data_processor.indices_dict['global_ind'][-1])
        # Returns a tuple containing the text of the lines with indices from first to last, inclusive. If the second argument is omitted, returns the text of the line closest to first.

        if self.DEFAULT:
            # update default plots

            # update data first
            self.dash.data_processor.update()

            # pull historical data
            self.full_data = self.dash.data_processor.data_dict_all

            # update default plot
            self.update_default_plot()

            # add data to selection box
            for client in self.full_data:
                # add variables to functionbox
                for varname in self.full_data[client]:
                    if varname == 'global_ind':
                        continue

                    box_val = r'{0}.{1}'.format(client, varname)
                    if box_val not in self.functionboxvals.get():
                        self.functionbox.insert(tk.END, box_val)

        self.set_timestamp_options()

    def plot_selected(self, selected_vals):
        # TAB 1 plotting procedure
        # TO DO:
        # - set limits properly
        # - behavior when multiple options selected for graphing, either from same or different clients
        self.update_status("Plotting selected values...")
        # clear old plot, prepare new subplot
        data = []
        self.figure_tab1.clf()
        a = self.figure_tab1.add_subplot(111)

        # Plotting procedure...
        # retrieve appropriate values from dicts
        for val in selected_vals:
            varlist = val.split(".")
            client = varlist[0]
            varname = '.'.join(varlist[1:])

            # size of array at each iteration

            if isinstance(self.full_data[client][varname], np.ndarray):
                if len(self.full_data[client][varname].shape) > 2:
                    size_in_second_dimension = np.prod((self.full_data[client][varname].shape)[1:])
                    plot_shape = (len(self.full_data[client]['global_ind']),) + (size_in_second_dimension,)

                    a.plot(self.full_data[client]['global_ind'],
                           self.full_data[client][varname].reshape(plot_shape),
                           label=varname)
                else:
                    a.plot(self.full_data[client]['global_ind'],
                           self.full_data[client][varname],
                           label=varname)
            else:
                print(self.full_data[client]['global_ind'], self.full_data[client][varname])
                a.plot(self.full_data[client]['global_ind'],
                       self.full_data[client][varname],
                       label=varname)

            if(self.labelval.get() == 1):
                a.set_xlabel('Iterations')
                a.set_ylabel('Selected Variable(s)')
            if(self.legendval.get() == 1):
                a.legend(loc='upper left', shadow=True, fontsize='small')
            # print(self.full_data[client][varname])

        # # plot data
        # try:
        #     a.plot(self.full_indices[client], data)
        # except:
        #     print("Indices don't match!")

        # actually draws updates on GUI
        self.canvas_tab1.draw()

    def update_default_plot(self):
        self.update_status("Updating default plot...")
        # Updates TAB 1 plots
        selected_plots = self.functionbox.curselection()
        values = []
        values.extend([self.functionbox.get(i) for i in selected_plots])
        self.plot_selected(values)

    def update_user_plots(self):
        """
        Callback for most recent frame button
        """
        self.update_status("Updating user plots...")
        # Updates TAB 2 plot
        self.view_iteration('most recent', 'GUI')

    def view_first(self):
        """
        Callback for first frame button
        """
        self.update_status("Showing first frame...")
        self.view_iteration([0], 'GUI')

    def view_slider_frame(self):
        """
        Callback for slider frame button
        """
        iter_num = self.slider_iteration.get()
        self.update_status('Showing slider frame ' + str(iter_num) + ' ...')
        self.view_iteration([iter_num], 'GUI')

    def view_iteration(self, interation_num, loc_type):
        """
        Most user plotting procedures go through here
        """
        if loc_type == 'GUI':
            self.dash.visualize(frame_ind=interation_num,
                                location_type=loc_type)
            self.canvas_tab2.draw()
        elif loc_type == 'standard_preview':
            self.dash.visualize(frame_ind=interation_num, show=True)
        elif loc_type == 'save_all':
            self.dash.visualize(frame_ind=interation_num)

    def preview_slider_frame(self):
        self.update_status("Generating preview...")
        iter_num = self.slider_iteration.get()
        self.view_iteration(iter_num, 'standard_preview')

    def save_all_frames(self):
        self.update_status("Saving frames...")
        self.view_iteration('all', 'save_all')
        self.update_status("Frames saved")

    def make_movie(self):
        self.update_status("Making movie...")
        self.dash.make_mov()
        self.update_status("Movie complete!")

    def onselect(self, evt, data_name):
        """
        Update current plot with selected data from listboxes.
        Also checks if the data is array-type and provides an
        additional listbox to select data within that array.
        """
        w = evt.widget
        values = [w.get(int(i)) for i in w.curselection()]
        self.update_default_plot()

    def auto_refresh(self):
        """
        automatically refreshes data
        """

        if self.auto_ref.get():
            self.root.after(1000, self.auto_refresh)
            self.refresh_data()

    def change_timestamp(self, *args):
        """
        Changes timestamp based on user selection
        """
        current_timestamp = string2string(
            full_timestamp=self.current_timestamp.get())

        self.dash.use_timestamp(
            date=current_timestamp[0], time=current_timestamp[1])

        self.update_status('Using Time ' + self.dash.timestamp_name + '...')

        self.refresh_data()

    def set_timestamp_options(self):
        """
        populates the optionbox for the timestamps
        """

        # OPtion menu items
        menu = self.timestamp_option['menu']

        # Get list of filenames in case_archive.
        ts_list = os.listdir(self.dash.case_archive_path)
        for timestamp_name in ts_list:
            if timestamp_name not in self.timestamp_list:
                menu.add_command(
                    label=timestamp_name, command=lambda timestamp_name=timestamp_name: self.current_timestamp.set(timestamp_name))
                self.timestamp_list.append(timestamp_name)

    def update_status(self, text):
        """
        Updates the text in the status box

        Parameters:
        -----------
            text: str
                string containing updated text
        """
        # Updates the string shown in the status bar
        status_list_new = [text]
        status_string = text + '\n...'
        for old_text in (self.status_list):
            status_list_new.append(old_text)

            status_string = old_text + '\n' + status_string

        # Writing to the status box
        self.statustext.set(status_string)

        # Storing old strings
        #self.string2 = string2
        self.status_list = status_list_new[:5]
        # for new_text in range(len(status_list_new)):
        #     self.status_list.append(new_text)

    def clear_selection(self):
        self.update_status('Clearing default plot selection...')
        self.functionbox.selection_clear(0, 'end')
        self.update_default_plot()

    def draw_GUI(self):
        print("draw_GUI")

        # -------------- SHARED OPTIONS: --------------
        sharedframelabel = ttk.Label(self.sharedframe, text="OPTIONS").grid(
            column=0, row=1)

        # System Status:
        self.statustext = tk.StringVar(self.root)
        self.update_status("Initializing...")
        statuslabel = ttk.Label(self.sharedframe, textvariable=self.statustext).grid(
            column=0, row=0, sticky='nw')
        # print(self.statustext.get())

        # Choose timestamp:
        timestamplabel = ttk.Label(self.sharedframe, text="Time Stamp:").grid(
            column=0, row=2, sticky='nesw')
        self.timestamp_option = tk.OptionMenu(
            self.sharedframe, self.current_timestamp, self.dash.timestamp_name)
        self.timestamp_option.grid(column=0, row=3, sticky='nesw')

        # auto refresh button
        auto_check = tk.Checkbutton(
            self.sharedframe, text="Auto-refresh", variable=self.auto_ref, command=self.auto_refresh, height=3)
        auto_check.grid(column=0, row=4, sticky='nesw')

        # refresh data button
        refreshbutton = tk.Button(
            self.sharedframe, text="Refresh data", command=self.refresh_data, height=3)
        refreshbutton.grid(column=0, row=5, sticky='nesw')

        # exit button
        quitbutton = tk.Button(
            self.sharedframe, text="Quit", command=self.quit, height=3)
        quitbutton.grid(column=0, row=6, sticky='nesw')

        # -------------- USER PLOT OPTIONS: --------------
        if self.USER:
            # add frame
            userFrame = ttk.Frame(self.frame_tab2)
            # userFrame.grid(row=2, column=0, sticky=(tk.W))
            userFrame.pack()

            userFrameTitle = ttk.Label(userFrame, text="USER VISUALIZATION OPTIONS").grid(
                column=0, row=0, sticky=(tk.W), columnspan=2)

            # SLIDER FRAME START----:
            # Frame
            slider_frame = ttk.Frame(
                userFrame, borderwidth=5, relief='ridge')
            slider_frame.grid(row=1, column=0, sticky=(tk.W), columnspan=2)

            # Slider
            self.iteration_slider = tk.Scale(slider_frame, orient="horizontal",
                                             from_=0, to=10,
                                             showvalue=True,
                                             length=500,
                                             variable=self.slider_iteration,
                                             tickinterval=100)
            self.iteration_slider.grid(
                column=0, row=1, columnspan=2, sticky='nesw')

            # Button
            slider_button = tk.Button(
                slider_frame,  text="Display Frame", command=self.view_slider_frame, height=3)
            slider_button.grid(column=0, row=2, sticky='nesw')

            preview_frames_button = tk.Button(
                slider_frame,  text="Preview frame to save", command=self.preview_slider_frame, height=3)
            preview_frames_button.grid(column=1, row=2, sticky='nesw')
            # SLIDER FRAME END----:

            view_first_button = tk.Button(
                userFrame,  text="Display First Frame", command=self.view_first, height=3)
            view_first_button.grid(column=0, row=2, sticky='nesw')

            view_mostrecent_button = tk.Button(
                userFrame,  text="Most Recent Frame", command=self.refresh_data, height=3)
            view_mostrecent_button.grid(column=1, row=2, sticky='nesw')

            save_frames_button = tk.Button(
                userFrame,  text="Save All Frames", command=self.save_all_frames, height=3)
            save_frames_button.grid(column=3, row=1, sticky='nesw')

            create_movie_button = tk.Button(
                userFrame,  text="Create Movie", command=self.make_movie, height=3)
            create_movie_button.grid(column=4, row=1, sticky='nesw')
        # -------------- DEFAULT PLOT OPTIONS: --------------

        if self.DEFAULT:
            # add frame
            boxFrame = ttk.Frame(self.frame_tab1)
            boxFrame.grid(row=1, column=0)
            boxFrameTitle = ttk.Label(boxFrame, text="Graphable Variables").grid(
                column=0, row=0, sticky=(tk.W, tk.E))

            self.functionbox = tk.Listbox(
                boxFrame, name="functionbox", selectmode=tk.MULTIPLE, width=30, exportselection=0, listvariable=self.functionboxvals
            )
            # self.functionbox = tk.Listbox(
            #     boxFrame, name="functionbox", selectmode=tk.MULTIPLE, exportselection=0, listvariable=self.functionboxvals
            # )
            self.functionbox.grid(column=0, row=1, sticky=(tk.W, tk.E))

            # add scrollbar
            boxscrollbar = tk.Scrollbar(boxFrame)
            boxscrollbar.grid(column=1, row=1, sticky=(tk.N, tk.S))
            self.functionbox.config(yscrollcommand=boxscrollbar.set)
            boxscrollbar.config(command=self.functionbox.yview)

            # add event listener
            self.functionbox.bind(
                "<<ListboxSelect>>", lambda event: self.onselect(event, self.full_data))

            # clear all selected variables
            clearbutton = tk.Button(boxFrame, text='Clear Selection',
                                    command=self.clear_selection)
            clearbutton.grid(column=0, row=2, sticky=(tk.W, tk.S))

            # graphing option list
            defaultOptionsFrame = ttk.Frame(boxFrame)
            defaultOptionsFrame.grid(row=1, column=2)

            # legend toggle
            self.legendval = tk.IntVar(self.root)
            self.legendtoggle = tk.Checkbutton(defaultOptionsFrame, text='Toggle Legend', justify='left', onvalue=1, offvalue=0, variable=self.legendval, command=self.update_default_plot)
            self.legendtoggle.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N))

            # axis label toggle
            self.labelval = tk.IntVar(self.root)
            self.labeltoggle = tk.Checkbutton(defaultOptionsFrame, text='Toggle Axis Labels', justify='left', onvalue=1, offvalue=0, variable=self.labelval, command=self.update_default_plot)
            self.labeltoggle.grid(column=0, row=1, sticky=(tk.W, tk.N))

        # plot initial
        self.current_timestamp.set(self.dash.timestamp_name)
        self.set_timestamp_options()

    # read pkl file, for each item in dictionary add to list so we can select what to plot

    def run_GUI(self):
        self.draw_GUI()
        tk.mainloop()

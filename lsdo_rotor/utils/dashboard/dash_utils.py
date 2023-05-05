import os
import importlib
import traceback
from datetime import datetime


working_dir_path = os.getcwd()
working_dir_path = working_dir_path.replace(r'\\ ', r'\ ')
working_dir_path = working_dir_path.replace(r'\ ', r' ')

DATETIME_FORMAT_STRING = '%Y-%m-%d-%H_%M_%S'
DATETIME_FORMAT_STRING_FILENAME = '%Y-%m-%d-%H_%M_%S_%f'
DATE_FORMAT_STRING = '%Y-%m-%d'
TIME_FORMAT_STRING = '%H_%M_%S'

WIDTH_GUI_PLOT = 10  # inches
HEIGHT_GUI_PLOT = 5  # inches

WIDTH_INTERFACE = 1400  # pixels
HEIGHT_INTERFACE = 800  # pixels


def get_dash(viz_file_name):
    try:
        # Get absolute file path of file containing the user dash class
        viz_file_path = r'{}/{}'.format(working_dir_path, viz_file_name)

        # Creates specifications/metadata from module
        spec = importlib.util.spec_from_file_location(
            'dash', viz_file_path)

        # takes above specifications and turns it into a module
        viz = importlib.util.module_from_spec(spec)

        # return instance
        spec.loader.exec_module(viz)
        print(viz.Dash())
        return viz.Dash()
    except:
        print(traceback.format_exc())
    return None


def get_abs_path(ts_path, dir_name):
    """
    This method returns the absolute path of "dir_name", assuming "dir_name" is within current working directory
    """
    full_path = r'{0}/{1}'.format(ts_path, dir_name)
    return full_path


def process_date_time(data_dir_path, date='', time=''):
    """
    Takes a user-specified date/time and returns the name of the folder ('YYYY-MM-DD-HH_MM_SS/')
    in the data directory closest.
    """
    # If one of either date or time is not specified, return error
    if date == '' and time != '':
        raise ValueError('Date not specified. Specify date or unspecify time')
    elif date != '' and time == '':
        raise ValueError('Time not specified. Specify time or unspecify date')

    # If date AND time are not specified, set time as current time
    if date == '' and time == '':
        user_date_time = datetime.now()
    else:
        # Parse date and time
        try:
            user_date_time = datetime.strptime(
                date+'-'+time, DATETIME_FORMAT_STRING)
        except:
            raise ValueError(
                'date or time is not in correct format. Use the following format: date = "YYYY-MM-DD" and time = "HH_MM_SS"')

    # If the data/frames directory does not exist at all, return
    print(data_dir_path)
    if not os.path.isdir(data_dir_path):
        return

    # Read in dates in each folder of data.
    data_date_time_dir_list = os.listdir(data_dir_path)

    data_date_time_list = []
    for date_time_dir in data_date_time_dir_list:
        try:
            data_date_time_list.append(datetime.strptime(
                date_time_dir, DATETIME_FORMAT_STRING))
        except:
            print('directory ', date_time_dir,
                  ' is not in a date/time format. Skipping directory. ')

    # Find closest date/time:
    closest_date_time = min(data_date_time_list,
                            key=lambda x: abs(x - user_date_time))

    # Get string
    closest_date_dir = date2string(closest_date_time)

    print(closest_date_dir)
    # Return string
    return closest_date_dir


def string2date(datetime='', date='', time=''):
    """
    create dateobject from string
    """
    if datetime == '':
        date_time_object = datetime.strptime(
            date+'-'+time, DATETIME_FORMAT_STRING)

    else:
        date_time_object = datetime.strptime(
            datetime, DATETIME_FORMAT_STRING)

    return date_time_object


def date2string(datetime_obj, separate=False):
    """
    create string from dateobject. Return two strings for dsate and time seperately if separate  == true. else concatenated string
    """
    if separate:
        return (datetime_obj.strftime(DATE_FORMAT_STRING), datetime_obj.strftime(TIME_FORMAT_STRING))
    else:
        return datetime_obj.strftime(DATETIME_FORMAT_STRING)


def string2string(full_timestamp='', date='', time=''):
    """
    concatenates or splits timestamp string
    """
    if full_timestamp == '':
        return date+'-'+time
    else:
        d = full_timestamp.split(
            '-')[0] + '-' + full_timestamp.split('-')[1] + '-' + full_timestamp.split('-')[2]
        t = full_timestamp.split('-')[3]
        return(d, t)


def create_file_name(*args):
    """
    Creates a file name given arguments.

    ex:
    create_file_name('test','file','png')
    returns 'test.file.png'

    Parameters:
    ----------
        args: str
            strings to create file name from.

    """

    # Initialize string
    filename = ''

    # Add names in args in order
    for i, name in enumerate(args):
        filename += str(name) + '.'

    # Do not return the final '.' character
    return filename[:-1]


def dir_exists(dir_path):
    """
    Given a file path, checks to see if the file exists. If so, return true. Otherwise return false.
    """
    # Check if filename exists
    if not os.path.isdir(dir_path):
        return False
    else:
        return True

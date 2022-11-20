import datetime


def shift_time_in_datetime_to_0(datetime_input):
    """Shifts time in datetime to 0

    >>> shift_time_in_datetime_to_0(datetime.datetime(2016, 12, 13, 11, 12, 13, 141516))
    datetime.datetime(2016, 12, 13, 0, 0)

    Parameters
    ----------
    datetime_input : `datetime.datetime`

    Returns
    -------
    datetime.datetime
         Datetime with time shifted to 0
    """
    return datetime.datetime(
        datetime_input.year, datetime_input.month, datetime_input.day
    )

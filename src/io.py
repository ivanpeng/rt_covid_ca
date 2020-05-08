"""
This file is a series of functions which should assist in IO operations to the file system.
"""
import json
import datetime
import requests


def write_dict_to_file(dict_obj, abs_path, date=datetime.date.today().strftime("%Y-%m-%d")):
    """
    Write a dict object to a path via JSON dumps
    :param dict_obj:
    :param abs_path:
    :param date:
    :return:
    """
    abs_path = '../export/data_by_day/rt-ca-{}.json'.format(date)
    with open(abs_path, 'w') as fp:
        json.dump({"columns": ["province", "date", "ML", "Low_90", "High_90"], "data": dict_obj}, fp)


def download_can_case_file(url="https://raw.githubusercontent.com/ishaberry/Covid19Canada/master/cases.csv",
                           filename="../data/linelist.csv"):
    # Download Canadian data file
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
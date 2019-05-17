"""
utils.py

This file holds a number of utility functions made available
particularly for a data related projects. It is safe to place
this file where the directory contains two folders named
data and results. These are used by this file to place data
in specific formats.

-Rakshit Agrawal, 2016
"""
import csv
import gzip
import os
import pickle
import random
import uuid
from csv import DictReader
from pprint import pprint
import json
import datetime
import numpy as np




class Utils:
    @staticmethod
    def data_file(filename):
        if not os.path.isdir(os.path.join(os.getcwd(), 'data')):
            os.mkdir(os.path.join(os.getcwd(), 'data'))
        return os.path.join(os.getcwd(), 'data', filename)

    @staticmethod
    def results_file(filename):
        if not os.path.isdir(os.path.join(os.getcwd(), 'results')):
            os.mkdir(os.path.join(os.getcwd(), 'results'))
        return os.path.join(os.getcwd(), 'results', filename)

    @staticmethod
    def save_to_pickle(data, picklename, use_results_dir=True):
        if use_results_dir:
            with open(Utils.results_file(picklename), 'wb') as outp:
                pickle.dump(data, outp)
        else:
            with open(picklename, 'wb') as outp:
                pickle.dump(data, outp)

    @staticmethod
    def save_to_json(data, json_name, use_results_dir=True):
        if use_results_dir:
            with open(Utils.results_file(json_name), 'wb') as outp:
                json.dump(data, outp)
        else:
            with open(json_name, 'wb') as outp:
                json.dump(data, outp)

    @staticmethod
    def load_list_from_file(filename):
        with open(filename, 'rb') as inp:
            data = inp.read()
            return data.splitlines()

    @staticmethod
    def load_from_pickle(picklename, type='results', use_abs_path=False):

        if use_abs_path:
            source_file = picklename
        elif type == 'results':
            source_file = Utils.results_file(picklename)
        elif type == 'data':
            source_file = Utils.data_file(picklename)
        else:
            raise ValueError

        with open(source_file, 'rb') as inp:
            return pickle.load(inp)

    @staticmethod
    def load_from_json(json_name, type='results', use_abs_path=False):

        if use_abs_path:
            source_file = json_name
        elif type == 'results':
            source_file = Utils.results_file(json_name)
        elif type == 'data':
            source_file = Utils.data_file(json_name)
        else:
            raise ValueError

        with open(source_file, 'rb') as inp:
            return json.load(inp)

    @staticmethod
    def push_list_of_delimiter_separated_strings_to_file(data_list, filename, delimiter=','):
        full_str = ""
        for i in data_list:
            line_str = delimiter.join([str(j) for j in i])
            full_str += line_str + "\n"

        with open(Utils.data_file(filename), 'wb') as outp:
            outp.write(full_str)

    @staticmethod
    def sha_hash_for_text(hash_base):
        return uuid.uuid5(uuid.NAMESPACE_OID, hash_base).hex

    @staticmethod
    def print_random(data_list):
        if len(data_list):
            pprint(data_list[random.randint(0, len(data_list))])
        else:
            print "No items found in the list"

    @staticmethod
    def n_random_samples_from_list(data_list, n=10):
        if not isinstance(data_list,list):
            return data_list
        if len(data_list) > n:
            return random.sample(data_list, n)
        else:
            return data_list

    @staticmethod
    def print_to_file(content, filename=None):
        if filename is None:
            filename = "outfile_at_{}".format(datetime.datetime.strftime(datetime.datetime.now(), '%Y%b%d_%H%M'))

        with open(Utils.results_file(filename), 'a') as outp:
            outp.write(content)

    @staticmethod
    def time_delta(t1, t2):
        return (t1 - t2).total_seconds()

    @staticmethod
    def get_date_range_for_n_days_before_date(last_date, n, date_format="%Y%m%d", return_as_datetime=False):
        if isinstance(last_date,datetime.datetime):
            base = last_date
        elif isinstance(last_date,str):
            base = datetime.datetime.strptime(last_date, date_format)
        else:
            raise ValueError('Date not identifiable')
        date_list = [base - datetime.timedelta(days=x) for x in range(0, n)]
        if return_as_datetime:
            return date_list
        else:
            return map(lambda x: datetime.datetime.strftime(x, date_format), date_list)

    @staticmethod
    def one_hot_vector_for_a_column_index(index, vector_size):
        index = int(index)
        vector_size = int(vector_size)
        a = np.zeros((vector_size))
        a[index] = 1
        return a

    @staticmethod
    def one_hot_vector_for_a_column_index_as_list(index, vector_size):
        return list(Utils.one_hot_vector_for_a_column_index(index, vector_size))

    @staticmethod
    def multi_hot_vector_for_index_list(index_list, vector_size):
        a = np.zeros(vector_size)
        for index in index_list:
            a[index] = 1
        return a

    @staticmethod
    def multi_hot_vector_for_index_list_as_list(index_list, vector_size):
        return list(Utils.multi_hot_vector_for_index_list(index_list, vector_size))

    @staticmethod
    def binary_representation_of_number_as_list(k, width=None):
        return [float(i) for i in np.binary_repr(k, width=width)]

    @staticmethod
    def binary_representation_of_number(k, width=None):
        return np.asarray(Utils.binary_representation_of_number_as_list(k, width))

    @staticmethod
    def yield_over_csv(filename, use_abs_path=False):
        if use_abs_path:
            file_to_use = filename
        else:
            file_to_use = Utils.data_file(filename)

        with open(file_to_use, 'r') as inp:
            csv_data = DictReader(inp)
            for entry in csv_data:
                yield entry

    @staticmethod
    def current_time_for_file_name():
        return datetime.datetime.strftime(datetime.datetime.now(), '%Y%b%d_%H%M')

    @staticmethod
    def write_list_to_gzip(data, filename, data_file=False, results_file=True, use_abs=False):
        if use_abs:
            filename = filename
        elif data_file:
            filename = Utils.data_file(filename)
        elif results_file:
            filename = Utils.results_file(filename)

        with gzip.open(filename, 'wb') as fp:
            for line in data:
                fp.write(json.dumps(line, default=Utils.datetime_serializer) + "\n")

        return filename

    @staticmethod
    def yield_from_gzip_jsons(filename, use_abs=False):
        if not use_abs:
            filename = Utils.data_file(filename)
        with gzip.open(filename, 'r') as data:
            for line in data:
                yield json.loads(line)

    @staticmethod
    def datetime_serializer(obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
            # else:
            #     json.JSONEncoder.default(obj)
            # # return obj

    @staticmethod
    def save_dict_to_csv(data, filename, use_results_dir=True):
        if use_results_dir:
            filename = Utils.results_file(filename)

        with open(filename, 'w') as outp:
            fieldnames = data[-1].keys()
            writer = csv.DictWriter(outp, fieldnames=fieldnames)
            writer.writeheader()
            for i in data:
                writer.writerow(i)

    @staticmethod
    def append_dict_to_csv(data, filename, use_results_dir=True):
        if use_results_dir:
            filename = Utils.results_file(filename)

        if not os.path.isfile(filename):
            write_mode = 'w'
        else:
            write_mode = 'a'

        with open(filename, write_mode) as outp:
            fieldnames = data[-1].keys()
            writer = csv.DictWriter(outp, fieldnames=fieldnames)
            if write_mode == 'w':
                writer.writeheader()
            for i in data:
                writer.writerow(i)

    @staticmethod
    def append_to_text_file(str, filename):
        with open(Utils.results_file(filename), 'a') as outp:
            outp.write("{}\n".format(str))

    @staticmethod
    def softmax(w, t = 1.0):
        e = np.exp(np.array(w) / t)
        dist = e / np.sum(e)
        return dist
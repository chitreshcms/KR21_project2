from typing import List

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, ticker


class PerfMetrics:
    def read_res(self) :
        print("reading for var from path;")
        varsizes = ["5", "10", "15", "20"]
        map_random_mean_runtimes_for_variable_size_dict = dict()
        map_minfill_mean_runtimes_for_variable_size_dict = dict()
        map_mindegree_mean_runtimes_for_variable_size_dict = dict()
        map_random_n_multiplication_for_variable_size_dict = dict()
        map_minfill_n_multiplication_for_variable_size_dict = dict()
        map_mindegree_n_multiplication_for_variable_size_dict = dict()

        mpe_random_mean_runtimes_for_variable_size_dict = dict()
        mpe_minfill_mean_runtimes_for_variable_size_dict = dict()
        mpe_mindegree_mean_runtimes_for_variable_size_dict = dict()
        mpe_random_n_multiplication_for_variable_size_dict = dict()
        mpe_minfill_n_multiplication_for_variable_size_dict = dict()
        mpe_mindegree_n_multiplication_for_variable_size_dict = dict()
        for vs in varsizes:
            v_mindegree_map = pd.read_csv("./VAR" + str(vs) + "_metrics_testmindegree_map.csv",
                                          names=["run_desc", "run_time", "n_mult"])
            v_mindegree_map_avg = np.mean(v_mindegree_map["run_time"])
            map_mindegree_mean_runtimes_for_variable_size_dict[vs] = v_mindegree_map_avg
            map_mindegree_n_multiplication_for_variable_size_dict[vs] = v_mindegree_map["n_mult"].max()


            v_random_map = pd.read_csv("./VAR" + str(vs) + "_metrics_testrandom_map.csv",
                                       names=["run_desc", "run_time", "n_mult"])
            v_random_map_avg = np.mean(v_random_map["run_time"])
            map_random_mean_runtimes_for_variable_size_dict[vs] = v_random_map_avg
            map_random_n_multiplication_for_variable_size_dict[vs] = v_random_map["n_mult"].max()


            v_minfill_map = pd.read_csv("./VAR" + str(vs) + "_metrics_testminfill_map.csv",
                                        names=["run_desc", "run_time", "n_mult"])
            v_minfill_map_avg = np.mean(v_minfill_map["run_time"])
            map_minfill_mean_runtimes_for_variable_size_dict[vs] = v_minfill_map_avg
            map_minfill_n_multiplication_for_variable_size_dict[vs] = v_minfill_map["n_mult"].max()


            v_mindegree_mpe = pd.read_csv("./VAR" + str(vs) + "_metrics_testmindegree_mpe.csv",
                                          names=["run_desc", "run_time", "n_mult"])
            v_mindegree_mpe_avg = np.mean(v_mindegree_mpe["run_time"])
            mpe_mindegree_mean_runtimes_for_variable_size_dict[vs] = v_mindegree_mpe_avg
            mpe_mindegree_n_multiplication_for_variable_size_dict[vs] = v_mindegree_mpe["n_mult"].max()


            v_random_mpe = pd.read_csv("./VAR" + str(vs) + "_metrics_testrandom_mpe.csv",
                                       names=["run_desc", "run_time", "n_mult"])
            v_random_mpe_avg = np.mean(v_random_mpe["run_time"])
            mpe_random_mean_runtimes_for_variable_size_dict[vs] = v_random_mpe_avg
            mpe_random_n_multiplication_for_variable_size_dict[vs] = v_random_mpe["n_mult"].max()

            v_minfill_mpe = pd.read_csv("./VAR" + str(vs) + "_metrics_testminfill_mpe.csv",
                                        names=["run_desc", "run_time", "n_mult"])
            v_minfill_mpe_avg = np.mean(v_minfill_mpe["run_time"])
            mpe_minfill_mean_runtimes_for_variable_size_dict[vs] = v_minfill_mpe_avg
            mpe_minfill_n_multiplication_for_variable_size_dict[vs] = v_minfill_mpe["n_mult"].max()

        # plt.yscale("log")
        # plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())
        # keys = map_random_mean_runtimes_for_variable_size_dict.keys()
        # values = map_random_mean_runtimes_for_variable_size_dict.values()

        random_map_X= map_random_mean_runtimes_for_variable_size_dict.keys()
        random_map_Y= map_random_mean_runtimes_for_variable_size_dict .values()
        # plt.plot(random_map_X, random_map_Y, linestyle="dashed", color='gray', label="random",marker='o')

        minfill_map_X= map_minfill_mean_runtimes_for_variable_size_dict .keys()
        minfill_map_Y= map_minfill_mean_runtimes_for_variable_size_dict .values()
        # plt.plot(minfill_map_X, minfill_map_Y, label="minfill",marker='x')

        mindegree_map_X= map_mindegree_mean_runtimes_for_variable_size_dict .keys()
        mindegree_map_Y= map_mindegree_mean_runtimes_for_variable_size_dict .values()
        # plt.plot(mindegree_map_X, mindegree_map_Y, label="mindegree", marker='.')

        random_mpe_X= mpe_random_mean_runtimes_for_variable_size_dict .keys()
        random_mpe_Y= mpe_random_mean_runtimes_for_variable_size_dict .values()
        # plt.plot(random_mpe_X, random_mpe_Y,  linestyle="dashed", color='gray', label="random",marker='o')

        minfill_mpe_X= mpe_minfill_mean_runtimes_for_variable_size_dict .keys()
        minfill_mpe_Y= mpe_minfill_mean_runtimes_for_variable_size_dict .values()
        # plt.plot(minfill_mpe_X, minfill_mpe_Y,  label="minfill",marker='x')

        mindegree_mpe_X= mpe_mindegree_mean_runtimes_for_variable_size_dict .keys()
        mindegree_mpe_Y= mpe_mindegree_mean_runtimes_for_variable_size_dict .values()
        # plt.plot(mindegree_mpe_X, mindegree_mpe_Y,  label="mindegree", marker='.')

        map_n_mult_randomX=map_random_n_multiplication_for_variable_size_dict.keys()
        map_n_mult_randomY=map_random_n_multiplication_for_variable_size_dict.values()
        map_n_mult_minfillX=map_minfill_n_multiplication_for_variable_size_dict.keys()
        map_n_mult_minfillY=map_minfill_n_multiplication_for_variable_size_dict.values()
        map_n_mult_mindegreeX=map_mindegree_n_multiplication_for_variable_size_dict.keys()
        map_n_mult_mindegreeY=map_mindegree_n_multiplication_for_variable_size_dict.values()

        w = 0.2


        mpe_n_mult_randomX= mpe_random_n_multiplication_for_variable_size_dict .keys()
        mpe_n_mult_randomY= mpe_random_n_multiplication_for_variable_size_dict .values()

        mpe_n_mult_minfillX=mpe_minfill_n_multiplication_for_variable_size_dict .keys()
        mpe_n_mult_minfillY=mpe_minfill_n_multiplication_for_variable_size_dict .values()

        mpe_n_mult_mindegreeX=mpe_mindegree_n_multiplication_for_variable_size_dict .keys()
        mpe_n_mult_mindegreeY=mpe_mindegree_n_multiplication_for_variable_size_dict .values()
        # br1 = np.arange(len(mpe_n_mult_randomX))
        br1 = np.arange(len(map_n_mult_randomX))
        br2 = [x + w for x in br1]
        br3 = [x + w for x in br2]
        plt.bar(br1, map_n_mult_randomY, label="random", width=w, )
        plt.bar(br2, map_n_mult_minfillY, label="minfill", width=w)
        plt.bar(br3, map_n_mult_mindegreeY, label="mindegree", width=w)
        # plt.bar(br1,mpe_n_mult_randomY, label="random",width=w,)
        # plt.bar(br2,mpe_n_mult_minfillY, label="minfill", width=w)
        # plt.bar(br3,mpe_n_mult_mindegreeY, label="mindegree",width=w)
        # plt.xticks([r + w for r in range(len(mpe_n_mult_randomX))],
        #            ['5', '10', '15', '20'])
        plt.xticks([r + w for r in range(len(map_n_mult_randomX))],
                   ['5', '10', '15', '20'])
        plt.xlabel(xlabel="Number of Variables")
        # plt.ylabel(ylabel="Runtime in ms (log)")
        plt.ylabel(ylabel="Number of CPT Multiplications")
        plt.grid(True)
        # plt.title(label="MAP Query with heuristics VS increasing variable size")
        # plt.title(label="MPE Query with heuristics VS increasing variable size")
        plt.title(label="Multiplications in MAP query for heuristics VS increasing variable size")
        # plt.title(label="Multiplications in MPE query for heuristics VS increasing variable size")
        plt.legend()
        plt.show()
        return []

    def start(self):
        self.read_res()

PerfMetrics().start()

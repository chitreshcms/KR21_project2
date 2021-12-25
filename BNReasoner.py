import copy
import csv
import time
from collections import deque
from random import random
from typing import Union, Tuple, List, Dict, Optional, Set

import networkx
import numpy as np
import pandas as pd
import pgmpy.readwrite
from pandas import DataFrame

import generator
from BayesNet import BayesNet


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        self.ordering = None
        self.marginal_cpt_for_all_vars = None
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
            # self.bn.load_from_bif(net)
            self.bn.draw_structure()
        else:
            self.bn = net
            # print("no args..")
            # g = generator.NetworkGenerator()
            # self.bn = g.generate_network(15)
        self.metadata = dict()
        self.metadata["leaf_nodes"] = []
        self.metadata["all_vars"] = self.bn.get_all_variables()
        self.nodes = dict()
        # self.evidence_set_true = ["dog-out"]
        import random
        # self.evidence_set_true = ["Rain?"]
        e1 = random.choice(self.bn.get_all_variables())
        e2 = random.choice(self.bn.get_all_variables())
        e3 = random.choice(self.bn.get_all_variables())
        e4 = random.choice(self.bn.get_all_variables())
        e5 = random.choice(self.bn.get_all_variables())
        # assert e1 != e2
        # self.evidence_set_true = [e1, e2, e3, e4, e5]
        # self.evidence_set_true = [e1, e2]
        self.evidence_set_true = []
        # self.evidence_set_true = [e1]
        # self.evidence_set_true = list(random.choice(list(self.bn.get_all_variables())))
        # self.evidence_set_false = [e3]
        self.evidence_set_false = []
        self.mult_count=0
        self.update_cpts_for_marginal_distributions(self.bn)
        # self.evidence_set_false = [""]

    # 1. d separation DONE
    # 2. ordering 2 done/ 1 left- TODO: minfill left
    # 3. network Pruning DONE
    # 4.  TODO:marginal distribution
    # 5.  TODO:map and mep
    # 7.  TODO:PERFORMANCE EVALUATION Show the comparative average performance of your implementation on the aforementioned tasks (MAP,
    # MPE) with different elimination order heuristics (min-order, min-fill vs. random order compared to one
    # another) w.r.t. increasing size of variables (growing with 10 more variables or more each time).
    # 1
    # by plots e.g., x-axis can time in seconds, while y-axis can be the number of variables.
    # Hint: You can of course create such big BNs manually, but automatic generation would make your
    # life much easier. This task will be graded according to the depth and elaboration of the analysis
    # 8.  TODO:Use case
    # •an a-priori marginal query.
    # •a posterior marginal query.
    # •one MAP and one MEP query.
    def print_metrics(self):
        # variables of problem (eg ; dog_problem- family-out,? dog-out? etc) :
        vars = self.bn.get_all_variables()
        for cpt in vars:
            print("\n\ncpt of :" + str(cpt))
            print("\n" + str(self.bn.get_cpt(cpt)))

            print("\n\nchildren of this variable :" + str(self.bn.get_children(cpt)))

            if (len(self.bn.get_children(cpt)) == 0):
                self.metadata["leaf_nodes"].append(cpt)
                print("leaf Node ")
            # print("\n" + str(self.bn.ge(cpt)))

    def detect_valves_in_path(self, variable_set_1, variable_set_2, evidence_set):

        vars = self.metadata["all_vars"]
        leaf_nodes = self.metadata["leaf_nodes"]
        for var in vars:
            if var in leaf_nodes:
                # we dont need to check for valve
                print("leafNode: " + var)
            isValve, whichValve = self.isValve(var)
            print("is Valve: " + var + " \nwhich Type:" + str(isValve) + " | " + str(whichValve))

    def get_leaf_nodes(self, bn: BayesNet):
        vars = bn.get_all_variables()
        leaf_nodes = []
        for var in vars:
            if len(bn.get_children(var)) < 1:
                leaf_nodes.append(var)
        return leaf_nodes

    def isValve(self, x) -> List[Union[bool, str]]:
        all_nodes = self.bn.get_all_variables()
        if x not in all_nodes:
            return [False, ""]
        if self.bn.structure.in_degree[x] == 2:
            print("convergent valve")
            return [True, "convergent"]
        elif self.bn.structure.out_degree[x] == 2:
            print("divergent valve")
            return [True, "divergent"]
        elif self.bn.structure.in_degree[x] == 1 and self.bn.structure.out_degree[x] == 1:
            print("sequential valve")
            return [True, "sequential"]
        else:
            print("no valve")
            return [False, ""]

    def get_minfill(self, bn: BayesNet):
        print("min fill ordering")
        # we need to eliminate that node that cause min neighbours
        all_vars = bn.get_all_variables()
        dict_factor_size = dict()
        for var in all_vars:
            dict_factor: dict = bn.get_interaction_graph().adj.get(var)
            dict_factor_size[var] = len(dict_factor.values())
            # print("debug err dict_factor_size for this var , is "+ str(var) + " and :" + str(dict_factor_size))
        # degree_list = sorted(list(dict_factor_size), key=lambda factor_dict: 0 if len(factor_dict[1])< 1 else factor_dict, reverse=False)
        degree_dict:List[List[str,int]] = [[]]
        for e,i in dict_factor_size.items():
            print("appending "+ e + str(i))
            degree_dict.append([e,i])
        print(str(dict_factor_size.items()))
        print(str(degree_dict))
        print(str(list((degree_dict))))
        degree_dict_r = degree_dict.remove([])

        fill_dict = sorted((degree_dict), key=lambda factor_dict_value: int(factor_dict_value[1]), reverse=False)
        degree_list=[]
        for ord_element in fill_dict:
            degree_list.append(ord_element[0])
        return degree_list

    def get_mindegree(self, bn):
        print("min degree ordering")
        ig_degree = bn.get_interaction_graph().degree
        all_vars = bn.get_all_variables()
        degree_dict = sorted(ig_degree, key=lambda degree_dict: degree_dict[1], reverse=False)
        degree_list = []
        for ord_element in degree_dict:
            degree_list.append(ord_element[0])

        return degree_list

    def get_random_ordering(self, bn):
        print("random ordering")
        ig_degree = bn.get_interaction_graph().degree
        import random
        # random.shuffle(ig_degree[1])
        degree_dict = ig_degree
        degree_list = []
        for ord_element in degree_dict:
            degree_list.append(ord_element[0])
        return degree_list

    def get_elimination_ordering(self, bn: BayesNet, heuristic: str):
        t_bn = self.bn
        if heuristic == "minfill":
            return self.get_minfill(bn)
        elif heuristic == "mindegree":
            return self.get_mindegree(t_bn)
        else:
            return self.get_random_ordering(bn)

    def d_separated(self, first_var_set: deque, second_var_set: deque, evidence_set: deque):
        dseparated = False
        temp_copy_net = copy.deepcopy(self.bn)
        while len(evidence_set) > 0:
            evidence = evidence_set.pop()
            edges = self.bn.structure.edges
            for edge in edges:
                if evidence in edge:
                    temp_copy_net.del_edge(edge)
            temp_copy_net.del_var(evidence)
            self.bn = temp_copy_net
            print("deleted variable ,edge : " + str(evidence) + "| " + str(edge))
        while len(first_var_set) > 0:
            start = first_var_set.pop()
            while (len(second_var_set) > 0):
                end = second_var_set.pop()
                print("start,end : " + str(start) + "," + str(end))

        return str(dseparated)

    def prune_bn(self, bn: BayesNet, first_var_set: deque, second_var_set: deque,
                 evidence_dict: Dict[str, bool]) -> BayesNet:
        union_set = first_var_set + second_var_set + deque(evidence_dict.keys())
        # if evidence set empty we just need to remove leaf notes that
        # don't belong to union set and no need to remove outgoing edges from evidence
        no_evidence = False
        if len(evidence_dict) == 0:
            no_evidence = True
        apply_prune_rules = True
        while apply_prune_rules:
            all_nodes = bn.get_all_variables()
            last_bn = bn
            for node in all_nodes:
                if node in self.get_leaf_nodes(last_bn):
                    # this is a leaf node
                    if node not in union_set:
                        # we can remove this node since not in union set
                        print("deleting var : " + str(node))
                        bn.del_var(node)
                    if not no_evidence:
                        if node in bn.structure.nodes:
                            outgoing_edges = []
                            for oe in bn.structure.out_edges:
                                if node in oe:
                                    outgoing_edges.append(oe)
                            # deleting all outgoing_edges
                            for out_edge in outgoing_edges:
                                print("deleting edge : " + str(out_edge))
                                if out_edge in bn.structure.nodes:
                                    cpt = bn.get_cpt(out_edge)
                                    bn.del_edge(out_edge)

                        for var, CPT in bn.get_all_cpts().items():
                            NEW_CPT = bn.get_compatible_instantiations_table(
                                instantiation=pd.Series(evidence_dict.keys()), cpt=CPT)
                            for ev_var in evidence_dict.keys():
                                if ev_var in NEW_CPT and ev_var != var:
                                    NEW_CPT = NEW_CPT.drop(ev_var, axis=1)
                                    bn.update_cpt(variable=var, cpt=NEW_CPT)
            if last_bn == bn:
                apply_prune_rules = False
        return bn

    def reduce_cpt(self, cpt: DataFrame, evidence_dict: Dict[str, bool]):
        res_cpt = copy.deepcopy(cpt)
        for e, tv in evidence_dict.items():
            if e not in cpt.columns:
                continue
            else:
                res_cpt = res_cpt[res_cpt[e] == tv]
                res_cpt = res_cpt.drop(columns=[e])
        return res_cpt

    def d_separation_with_pruning(self, first_var_set: deque, second_var_set: deque, evidence_set: deque) -> bool:
        d_sep = False
        # first we apply pruning rules
        pruned_bn = self.prune_bn(self.bn, first_var_set, second_var_set, evidence_set)

        # now if there is no path for each start in first-var-set to end in second-var-set then they are dseparated
        keep_searching = True
        while len(first_var_set) > 0 and keep_searching:
            start = first_var_set.pop()
            while len(second_var_set) > 0 and keep_searching:
                end = second_var_set.pop()
                path_exist = False
                path_exist = pruned_bn.structure.has_successor(start, end)
                path_exist = path_exist or pruned_bn.structure.has_predecessor(start, end)
                if not path_exist:
                    print("no path exist for : start,end : " + str(start) + ", " + str(end))
                    d_sep = False
            # if we couldnot find any path
        return d_sep

    def prune_bn_for_query(self, query_set: deque, evidence_dict: Dict[str, bool], bn: BayesNet):
        empty_set = deque()
        pruned_bn = self.prune_bn(bn, query_set, empty_set, evidence_dict)
        return pruned_bn

    def print_cpts(self):
        S = self.bn.get_all_cpts()
        factors = {}
        print([S[cpt] for cpt in [_ for _ in S]], "\n=======================================\n")

    def update_cpts_for_marginal_distributions(self, bn: BayesNet):
        print("var len of bn is " + str(len(bn.get_all_variables())))
        self.marginal_cpt_for_all_vars = bn.get_all_cpts()

    def apply_evidence_and_zero_out_rows(self, cpts, evidence_variables: Dict[str, bool]):
        res_cpts = copy.deepcopy(cpts)
        for evidence in evidence_variables.items():
            for res_var, res_cpt in res_cpts.items():
                if evidence[0] in res_cpt.columns:
                    for idx, row in res_cpt.iterrows():
                        if res_cpt.loc[idx, evidence[0]] != evidence[1]:
                            res_cpt.loc[idx, 'p'] = 0
                            # print("evidence in this cpt.." + str(evidence[0]) + " set at location " + str(idx))
                res_cpts[evidence[0]] = res_cpt
        return res_cpts

    def get_evidence_dict(self, evidence_set_true, evidence_set_false):
        e_dict: Dict[str, bool] = dict()
        for var in self.bn.get_all_variables():
            if var in evidence_set_true:
                e_dict[var] = True
            elif var in evidence_set_false:
                e_dict[var] = False
            else:
                continue
        return e_dict

    def sum_out_and_eliminate_variable(self, result: DataFrame, var) -> DataFrame:
        if var not in result.columns:
            return result
        else:
            # added condition to avoid summing out single var from result
            if len(result.columns.difference(["p"])) == 1:
                return result
            grouped_result = result.drop([var], axis=1)
            grouped_result = grouped_result.groupby(list(grouped_result.columns.difference(["p"]))).agg(
                "sum").reset_index()
            return grouped_result

    def max_out_and_eliminate_variable(self, df: DataFrame, var):
        if var not in df.columns:
            return df
        else:
            if len(df.columns.difference(["p"])) == 1:
                return df
            grouped_result = df.drop([var], axis=1)
            grouped_result = grouped_result.groupby(list(grouped_result.columns.difference(["p"]))).agg(
                "max").reset_index()
            return grouped_result

    def variable_elimination(self, cpts, for_mpe=False):
        # print("debug marker variable elimination:" + str(""))

        if self.ordering is not None:
            waiting_vars_to_be_eliminated = deque(reversed(self.ordering))
        else:
            waiting_vars_to_be_eliminated = deque(self.get_mindegree(self.bn))
        updated_cpts_after_elimination: List[DataFrame] = []
        while len(waiting_vars_to_be_eliminated) > 0:
            var = waiting_vars_to_be_eliminated.pop()
            cpts_for_multiplication: List[DataFrame] = []
            for cpt in cpts.items():
                if var in cpt[1].columns:
                    cpts_for_multiplication.append(cpt[1])
                    result = cpts_for_multiplication[0]
                    for c in cpts_for_multiplication:
                        result = self.multiply_cpts(c, result)
                        # print("debug marker multiplied cpts for var" + str(var))
                    # This is the part where we will sum out to eliminate the variable
                    if for_mpe:
                        maxed_out_result = self.max_out_and_eliminate_variable(result, var)
                        updated_cpts_after_elimination.append(maxed_out_result)
                    else:
                        summed_out_result = self.sum_out_and_eliminate_variable(result, var)
                        updated_cpts_after_elimination.append(summed_out_result)
        return updated_cpts_after_elimination

    def get_merged_result_cpt(self, cpt1, cpt2):
        # t1 = time.time()
        cols1 = cpt1.columns.difference(["p"])
        cols2 = cpt2.columns.difference(["p"])
        cols_to_add = cols2.difference(cols1)
        merged_cpt = copy.deepcopy(cpt1)
        # increase the table by adding the difference
        for col in cols_to_add:
            insert_idx = len(cols1) - 1
            if {True, False}.issubset(set(cpt2[col])):
                old = copy.deepcopy(merged_cpt)
                merged_cpt.insert(insert_idx, col, True)
                merged_cpt = pd.concat([merged_cpt, old]).fillna(False)
            else:
                for tv in [True, False]:
                    merged_cpt.insert(insert_idx, col, tv)
            merged_cpt = merged_cpt.sort_values(by=list(merged_cpt.columns)).reset_index(drop=True)
        # t2 = time.time()
        # print("metric : time taken for mergedresult :" + str(t2 - t1) + "ms")

        return merged_cpt
    def get_cv(self, cpt1,cpt2):
        cv = list(
            set([col for col in cpt1.columns if col != 'p']) & set([col for col in cpt2.columns if col != 'p']))
        return cv
    def multiply_cpts(self, cpt1, cpt2):
        cpt1 = copy.deepcopy(cpt1)
        cpt2 = copy.deepcopy(cpt2)
        if len(cpt1)<1:
            return cpt2
        elif len(cpt2)<1:
            return cpt1
        if type(cpt1) is pd.Series:
            cpt1 = cpt1.to_frame().T
        if type(cpt2) is pd.Series:
            cpt2 = cpt2.to_frame().T
        cv= self.get_cv(cpt1,cpt2)
        if not cv:
            return pd.merge(cpt1, cpt2, on=['p'])

        print("mergeing cpt1 n cpt2 on " + str(cv))
        merged_df = pd.merge(cpt1, cpt2, on=cv)
        merged_df['p'] = (merged_df['p_1'] * merged_df['p_2'])
        merged_df.drop(['p_1', 'p_2'], inplace=True, axis=1)
        self.mult_count = self.mult_count + 1
        return merged_df

    def get_evidence_deque_from_dict(self, evidence_dict: Dict[str, bool]) -> deque:
        if len(evidence_dict) < 1:
            return deque()
        else:
            e_list = deque()
            for e, v in evidence_dict.items():
                e_list.append(e)
            return e_list

    def get_marginal_distribution(self, query_variables: deque, evidence_variables: Dict[str, bool]):
        print("getting marginal distributions for " + str(query_variables) + "\n given evidence set is : " + str(
            evidence_variables))
        # t1 = time.time()
        self.bn = self.prune_bn_for_query(query_variables, evidence_variables, self.bn)
        # t2 = time.time()
        # print("metric : time taken for pruning :" + str(t2 - t1) + "ms")

        zeroed_out_cpts = self.apply_evidence_and_zero_out_rows(self.marginal_cpt_for_all_vars, evidence_variables)
        # t1 = time.time()
        updated_cpts_after_elimination = self.variable_elimination(zeroed_out_cpts)
        # t2 = time.time()
        # print("metric : time taken for variable elimination :" + str(t2 - t1) + "ms")

        final_cpt = updated_cpts_after_elimination.pop(0)
        for cpt in updated_cpts_after_elimination:
            # print("debug marker multiplying for final cpt;;; " + str(cpt))
            final_cpt = self.multiply_cpts(cpt, final_cpt)
        # final_cpt= self.get_evidence_denominator()
        if len(evidence_variables) > 0:
            denom =1
            evidence_marginals = self.get_marginal_distribution((self.get_evidence_deque_from_dict(evidence_variables)),
                                                                {})
            for i, evidence_row in evidence_marginals.iterrows():
                denom = 1
                for evidence_var, evidence_value in evidence_row.items():
                    if evidence_var != 'p' and [str(evidence_var)] != evidence_value:
                        break
                    else:
                        if evidence_var == "p":
                            denom = evidence_value
            # for idx, row in final_cpt.iterrows():
            #     final_cpt.loc[idx, 'p'] /= denom
            final_cpt['p'] = final_cpt['p'] / denom

        return final_cpt

    def run_map_query(self, query_variables: deque, evidence_variables: Dict[str, bool]) -> pd.DataFrame:
        md = self.get_marginal_distribution(query_variables, evidence_variables)
        if (md['p'].count()) > 0:
            print("MAP :\n" + str(md.iloc[md['p'].idxmax()]))
            return md.iloc[md['p'].idxmax()]
        else:
            print("MAP: Fatal error")
            return pd.DataFrame()

    def get_partial_marginal_distribution(self, query_variables: deque, evidence_variables: Dict[str, bool]):
        self.bn = self.prune_bn_for_query(query_variables, evidence_variables, self.bn)

        zeroed_out_cpts = self.apply_evidence_and_zero_out_rows(self.marginal_cpt_for_all_vars, evidence_variables)
        updated_cpts_after_elimination = self.variable_elimination(zeroed_out_cpts, for_mpe=True)
        final_cpt = updated_cpts_after_elimination.pop(0)
        for cpt in updated_cpts_after_elimination:
            final_cpt = self.multiply_cpts(cpt, final_cpt)
        # final_cpt= self.get_evidence_denominator()
        if len(evidence_variables) > 0:
            denom =1
            evidence_marginals = self.get_marginal_distribution((self.get_evidence_deque_from_dict(evidence_variables)),
                                                                {})
            for i, evidence_row in evidence_marginals.iterrows():
                denom = 1
                for evidence_var, evidence_value in evidence_row.items():
                    if evidence_var != 'p' and [str(evidence_var)] != evidence_value:
                        break
                    else:
                        if evidence_var == "p":
                            denom = evidence_value
            # for idx, row in final_cpt.iterrows():
            #     final_cpt.loc[idx, 'p'] /= denom
            final_cpt['p'] = final_cpt['p'] / denom
        return final_cpt

    def run_mpe_query(self, query_variables: deque, evidence_variables: Dict[str, bool]) -> pd.DataFrame:
        # query_variables = deque(self.bn.get_all_variables())
        pmd = self.get_partial_marginal_distribution(query_variables, evidence_variables)
        print("MPE :\n" + str(pmd.iloc[pmd['p'].idxmax()]))
        return pmd.iloc[pmd['p'].idxmax()]

    def start(self):
        print("S")
        self.print_metrics()
        print("number of leafNodes:" + str(len(self.metadata["leaf_nodes"])))
        print(self.bn.get_interaction_graph().nodes)
        print(self.bn.get_interaction_graph().edges)
        print(self.bn.get_interaction_graph().adj)
        print(self.bn.get_interaction_graph().degree)
        evidence = ["light-on", "bowel-problem"]
        for node in self.bn.get_interaction_graph().nodes:
            if node in evidence:
                print("node in evidence" + node)

        test_d_sep_X = ["dog-out"]
        test_d_sep_Y = ["family-out"]
        # test_d_sep_evidence= ["bowel-problem"]
        test_d_sep_evidence = []
        # print("checking d sep for " + str(test_d_sep_X) + str(test_d_sep_Y) + str(test_d_sep_evidence))
        # d_sep = self.d_separation_with_pruning(deque(test_d_sep_X), deque(test_d_sep_Y), deque(test_d_sep_evidence))
        # self.prune_bn_for_query(deque(test_d_sep_X), deque(test_d_sep_evidence))
        # print(d_sep)
        mindegree = self.get_mindegree(self.bn)
        minfill = self.get_minfill(self.bn)
        random_odering = self.get_random_ordering(self.bn)
        for mf in minfill:
            print("->" + str(mf))
        print("\n")
        for md in mindegree:
            print("->" + str(md))
        print("\n")
        for ro in random_odering:
            print("->" + str(ro))
        print("\n")
        evidence_dict: Dict[str, bool] = dict()

        # for var in self.metadata["all_vars"]:
        #     if var in evidence:
        #         evidence_dict[var] = True
        #     else:
        #         evidence_dict[var] = False
        ord_tuple: Tuple[str, int] = tuple()
        index = 0
        for var in minfill:
            ord_tuple += tuple([var, index])
            index = index + 1

        # self.marginal_distributions(list(self.metadata["all_vars"]),evidence_dict,ord_tuple)
        # self.compute_marginal_distribution(set(list(self.metadata["all_vars"])),evidence_dict)
        self.marginal_cpt_for_all_vars: Dict[str, DataFrame] = self.bn.get_all_cpts()

        self.get_marginal_distribution(deque(self.metadata["all_vars"]),
                                       self.get_evidence_dict(self.evidence_set_true, self.evidence_set_false))
        self.run_map_query(deque(set(self.metadata["all_vars"]).difference({"bowel-problem"})),
                           self.get_evidence_dict(self.evidence_set_true, self.evidence_set_false))

        self.run_mpe_query(deque(self.metadata["all_vars"]),
                           self.get_evidence_dict(self.evidence_set_true, self.evidence_set_false))
        """
        min=self.get_random_ordering(self.bn)

        # self.bn = self.prune_bn(self.bn, deque(["dog-out", "family-out"]), deque(["bowel-problem"]),
        #                         deque(["hear-bark"]))
        # print("cms_d_sep"+str( self.d_separation(["dog-out"],["bowel-problem"],[])))
        # print("cms_d_sep" + str(self.d_separated(deque(["dog-out"]), deque(["bowel-problem"]), deque(["family-out"]))))
        # for node in self.bn.get_all_variables():
        #     print("isValve :" + node)
        #     self.isValve(node)
        # all_vars_start = deque(self.bn.get_all_variables())
        # all_vars_end = deque(self.bn.get_all_variables())
        # while len(all_vars_start) > 0 and len(all_vars_end) > 0:
        #     start = all_vars_start.pop()
        #     copy_end_vars = all_vars_end.copy()
        #     while len(copy_end_vars) > 0:
        #         end = copy_end_vars.pop()
        #         print("detecting valves in path from " + start + " to " + end + ":-")
        #         self.detect_valves_in_path(start, end, set())
        
        """
        self.bn.draw_structure()
    def run_queries(self):
        print("running query :")
        self.marginal_cpt_for_all_vars: Dict[str, DataFrame] = self.bn.get_all_cpts()
        self.ordering= self.get_elimination_ordering(self.bn,"mindegree")
        qv1="no_groups_meeting?"
        qv2="late_submission?"
        qv3="individual_submission?"
        self.query_vars = deque([qv1, qv2, qv3])
        # self.query_vars = deque([qv3])
        self.evidence_set_true=["pass?"]
        # self.evidence_set_true=["most_lectures_attended? "]
        # self.evidence_set_true=["project_completed?","most_lectures_attended?"]
        # self.evidence_set_false=["sudden_emergency?","individual_submission?"]
        self.evidence_set_false=["sudden_emergency?"]
        # self.evidence_set_false=[]
        # res = self.get_marginal_distribution(self.query_vars,self.get_evidence_dict(self.evidence_set_true,self.evidence_set_false)
# )
#         print("ans : md " + str(res.iloc[res['p'].idxmax()]))
#         self.run_map_query(self.query_vars, self.get_evidence_dict(self.evidence_set_true, self.evidence_set_false))
        self.run_mpe_query(self.query_vars, self.get_evidence_dict(self.evidence_set_true, self.evidence_set_false))

    def metric_run(self):
        self.metadata["metric_num_vars"] = [str(len(self.bn.get_all_variables()))]
        self.metadata["metric_network_size_n_nodes"] = [str(len(self.bn.get_interaction_graph().nodes))]
        export_current_run_metric = dict()

        # self.query_vars= deque(["Rain?","Sprinkler?"])
        self.marginal_cpt_for_all_vars: Dict[str, DataFrame] = self.bn.get_all_cpts()

        # self.ordering = self.get_elimination_ordering(self.bn, "minfill")
        self.ordering= self.get_elimination_ordering(self.bn,"mindegree")
        # self.ordering = self.get_elimination_ordering(self.bn, "random")
        runtimes = dict()
        s_time = time.time()
        self.backup_bn = self.bn

        if self.metadata.__contains__("metric_n_runs"):
            n = self.metadata["metric_n_runs"]
        else:
            # default run 10 times each query
            # n=1
            n = 10
        for i in range(1, n + 1):
            import random
            self.bn = self.backup_bn
            self.mult_count =0
            tall_vars = self.bn.get_all_variables()
            qv1 = random.choice(tall_vars)
            nv =[]
            for tv in tall_vars:
                if qv1 != tv:
                    nv.append(tv)

            qv2 = random.choice(nv)
            qv3 = random.choice(nv)
            print(str(qv1)+str(qv2))
            assert qv1 != qv2
            # self.query_vars = deque([qv1, qv2])
            self.query_vars = deque([qv1,qv2,qv3])
            # s_time = time.time()
            # self.run_map_query(self.query_vars, self.get_evidence_dict(self.evidence_set_true, self.evidence_set_false))
            # e_time = time.time()

            s_time = time.time()
            self.run_mpe_query(self.query_vars,self.get_evidence_dict(self.evidence_set_true,self.evidence_set_false))
            e_time = time.time()

            rt = float(e_time - s_time)
            # runtimes["random"+"_map_"+str(i)]= [rt,self.mult_count]
            # runtimes["minfill"+"_map_"+str(i)]= [rt,self.mult_count]
            # runtimes["mindegree"+"_map_"+str(i)]= [rt,self.mult_count]
            # runtimes["random"+"_mpe_"+str(i)]= [rt,self.mult_count]
            # runtimes["minfill"+"_mpe_"+str(i)]= [rt,self.mult_count]
            runtimes["mindegree"+"_mpe_"+str(i)]= [rt,self.mult_count]
            # self.run_map_query(self.query_vars,self.get_evidence_dict(self.evidence_set_true,self.evidence_set_false))
        # e_time = time.time()
        # rt = float(e_time - s_time)
        # print("time taken : " + str(rt))

        # a_file = open("VAR5_metrics_test"+"random_map" +".csv", "w")
        # a_file = open("VAR5_metrics_test"+"minfill_map" +".csv", "w")
        # a_file = open("VAR5_metrics_test"+"mindegree_map" +".csv", "w")
        # a_file = open("VAR5_metrics_test"+"random_mpe" +".csv", "w")
        # a_file = open("VAR5_metrics_test"+"minfill_mpe" +".csv", "w")
        # a_file = open("VAR5_metrics_test"+"mindegree_mpe" +".csv", "w")

        # a_file = open("VAR10_metrics_test"+"random_map" +".csv", "w")
        # a_file = open("VAR10_metrics_test"+"minfill_map" +".csv", "w")
        # a_file = open("VAR10_metrics_test"+"mindegree_map" +".csv", "w")
        # a_file = open("VAR10_metrics_test"+"random_mpe" +".csv", "w")
        # a_file = open("VAR10_metrics_test"+"minfill_mpe" +".csv", "w")
        # a_file = open("VAR10_metrics_test"+"mindegree_mpe" +".csv", "w")

        # a_file = open("VAR15_metrics_test"+"random_map" +".csv", "w")
        # a_file = open("VAR15_metrics_test"+"minfill_map" +".csv", "w")
        # a_file = open("VAR15_metrics_test"+"mindegree_map" +".csv", "w")
        # a_file = open("VAR15_metrics_test"+"random_mpe" +".csv", "w")
        # a_file = open("VAR15_metrics_test"+"minfill_mpe" +".csv", "w")
        a_file = open("VAR15_metrics_test"+"mindegree_mpe" +".csv", "w")

        # a_file = open("VAR20_metrics_test"+"random_map" +".csv", "w")
        # a_file = open("VAR20_metrics_test"+"minfill_map" +".csv", "w")
        # a_file = open("VAR20_metrics_test"+"mindegree_map" +".csv", "w")
        # a_file = open("VAR20_metrics_test"+"random_mpe" +".csv", "w")
        # a_file = open("VAR20_metrics_test"+"minfill_mpe" +".csv", "w")
        # a_file = open("VAR20_metrics_test"+"mindegree_mpe" +".csv", "w")

        # a_file = open("VAR25_metric_test"+"random_map" +".csv", "w")
        # a_file = open("VAR25_metric_test"+"minfill_map" +".csv", "w")
        # a_file = open("VAR25_metric_test"+"mindegree_map" +".csv", "w")
        # a_file = open("VAR25_metric_test"+"random_mpe" +".csv", "w")
        # a_file = open("VAR25_metric_test"+"minfill_mpe" +".csv", "w")
        # a_file = open("VAR25_metric_test"+"mindegree_mpe" +".csv", "w")

        # a_file = open("VAR35_metric_test"+"random_map" +".csv", "w")
        # a_file = open("VAR35_metric_test"+"minfill_map" +".csv", "w")
        # a_file = open("VAR35_metric_test"+"mindegree_map" +".csv", "w")
        # a_file = open("VAR35_metric_test"+"random_mpe" +".csv", "w")
        # a_file = open("VAR35_metric_test"+"minfill_mpe" +".csv", "w")
        # a_file = open("VAR35_metric_test"+"mindegree_mpe" +".csv", "w")

        # a_file = open("VAR45_metric_test"+"random_map" +".csv", "w")
        # a_file = open("VAR45_metric_test"+"minfill_map" +".csv", "w")
        # a_file = open("VAR45_metric_test"+"mindegree_map" +".csv", "w")
        # a_file = open("VAR45_metric_test"+"random_mpe" +".csv", "w")
        # a_file = open("VAR45_metric_test"+"minfill_mpe" +".csv", "w")
        # a_file = open("VAR45_metric_test"+"mindegree_mpe" +".csv", "w")

        # a_file = open("VAR55_metric_test"+"random_map" +".csv", "w")
        # a_file = open("VAR55_metric_test"+"minfill_map" +".csv", "w")
        # a_file = open("VAR55_metric_test"+"mindegree_map" +".csv", "w")
        # a_file = open("VAR55_metric_test"+"random_mpe" +".csv", "w")
        # a_file = open("VAR55_metric_test"+"minfill_mpe" +".csv", "w")
        # a_file = open("VAR55_metric_test"+"mindegree_mpe" +".csv", "w")

        writer = csv.writer(a_file)
        for key, value in runtimes.items():
            writer.writerow([key, value[0], value[1]])

        a_file.close()
        print(runtimes)

# bnr = BNReasoner("testing/dog_problem.BIFXML")
# bnr = BNReasoner("testing/lecture_example.BIFXML")
# bnr = BNReasoner("testing/b40-51.xml")
# bnr = BNReasoner("./cms_size15.BIFXML")
bnr = BNReasoner("testing/group_1_use_case.BIFXML")
# bnr.start()
# bnr = BNReasoner(["",None])
# varSize = 5
# varSize = 10
varSize = 15
# varSize = 20
# varSize = 25
# varSize = 30
# bng = generator.BNGenerator(varSize)
# v5 = bng.bayes_net
# bnr = BNReasoner(v5)
# bnr.metric_run()
# bnr.final_metric_run()
# bnr.run_queries()
cpts_file = open("CMS_USE_CASE_CPTs.csv", "w")
it=0
for v,cpt in bnr.bn.get_all_cpts().items():
    print(v)
    print(cpt)
    cpt.to_csv("var_"+v+str(it)+"_Cpt.csv")
    it= it+1
    # writer = csv.writer(cpts_file)
    # writer.writerow([v, cpt])
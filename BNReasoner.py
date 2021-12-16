from typing import Union
from BayesNet import BayesNet


class BNReasoner:
    def __init__(self, net: Union[str, BayesNet]):
        """
        :param net: either file path of the bayesian network in BIFXML format or BayesNet object
        """
        if type(net) == str:
            # constructs a BN object
            self.bn = BayesNet()
            # Loads the BN from an BIFXML file
            self.bn.load_from_bifxml(net)
            # self.bn.draw_structure()
        else:
            self.bn = net
            print("no args..")
    #1. d separation
    #2. ordering
    #3. network Pruning
    #4. marginal distribution
    #5. map and mep
    #7. PERFORMANCE EVALUATION Show the comparative average performance of your implementation on the aforementioned tasks (MAP,
            # MPE) with different elimination order heuristics (min-order, min-fill vs. random order compared to one
            # another) w.r.t. increasing size of variables (growing with 10 more variables or more each time).
            # 1
            # by plots e.g., x-axis can time in seconds, while y-axis can be the number of variables.
            # Hint: You can of course create such big BNs manually, but automatic generation would make your
            # life much easier. This task will be graded according to the depth and elaboration of the analysis
    #8. Use case
        #•an a-priori marginal query.
        # •a posterior marginal query.
        # •one MAP and one MEP query.
    def print_metrics(self):
        #variables of problem (eg ; dog_problem- family-out,? dog-out? etc) :
        vars =self.bn.get_all_variables()
        for cpt in vars:
            print("\n\ncpt of :" + str(cpt))
            print("\n" + str(self.bn.get_cpt(cpt)))
    def start(self) :
        print("S")
        self.print_metrics()
bnr = BNReasoner("testing/dog_problem.BIFXML")
bnr.start()
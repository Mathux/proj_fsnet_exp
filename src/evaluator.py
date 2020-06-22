import numpy as np
from sklearn import svm
from src.tools import create_dir
import os
import json


class Experiments:
    def __init__(self, dataset):
        self.dataset = dataset
        self.experiments = {}
        
    def add_experiment(self, seed, data, models):
        scores = {}
        times = {}
        for modelname, model in models.items():
            scores_model = {}
            score_regular = self.svm_score(data, model,
                                           balanced=False,
                                           seed=seed)
            score_balanced = self.svm_score(data, model,
                                            balanced=True,
                                            seed=seed)
            scores_model = {"regular": score_regular,
                            "balanced": score_balanced}
            scores[modelname] = scores_model
            times[modelname] = model.time_

        self.experiments[seed] = {"scores": scores, "times": times}
        
    def save(self, folder=None):
        if folder is None:
            folder = "{}_experiments".format(self.dataset)
            
        create_dir(folder)
        expname = os.path.join(folder, "{}_experiments.json".format(self.dataset))
        with open(expname, "w") as expfile:
            expfile.write(json.dumps(self.experiments, indent=True))
        
    def create_figure(self, folder=None, show=True):
        if folder is None:
            folder = "{}_experiments".format(self.dataset)
                        
        create_dir(folder)

        for balanced in ["balanced"]:  # , "regular"]:
            for frots in [False]: # [True, False]:
                kargs = {"balanced": balanced,
                         "frots": frots}
                figname = os.path.join(folder, "{}_feature_selection.png")
                figname = figname.format(self.dataset)
                # , "frot" if frots else "allmethods", balanced)
                
                # Average scores and create a figure
                self.create_scores_graph_fig(figname, **kargs)

        self.show_time()

    def show_time(self, frots=False):
        dico = self.experiments

        seeds = list(dico)
        methods = list(dico[seeds[0]]["scores"])
        
        time_means = {}
        for method in methods:
            times = [dico[seed]["times"][method] for seed in seeds]
            meantime = np.around(np.mean(times), 2)
            intervaltime = np.around(1.96*np.std(times), 2)
            
            if frots:
                if not method.startswith("FROT"):
                    continue
            else:
                if method.startswith("FROT"):
                    eta = float(method.split("_")[1])
                    if not eta == 1.0:
                        continue
                    method = "FROT"
                        
            time_means[method] = '{} ($\pm$ {})'.format(meantime, intervaltime)

        ptimes = [time_means[method] for method in ["Wasserstein", "Linear correlation", "MMD", "FROT"]]
        print("{} & d & n & {} & {} & {} & {} \\\\".format(self.dataset.capitalize(), *ptimes))
        
    def svm_score(self, data, model, balanced=True, seed=42):
        if balanced:
            clf = svm.SVC(gamma="scale", class_weight="balanced", random_state=seed)
        else:
            clf = svm.SVC(gamma="scale", random_state=seed)
        train_gfeats = model.transform(data.X_train)
        clf.fit(train_gfeats, data.y_train)

        test_gfeats = model.transform(data.X_test)        
        score = clf.score(test_gfeats, data.y_test)
        return score

    def load_experiments(self, exp):
        self.experiments = json.load(open(exp))
    
    def create_scores_graph_fig(self, figname, frots=True, balanced="balanced"):
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 16})
        
        ngroups = np.arange(10, 51, 10)
        groups = [str(x) for x in ngroups]

        dico = self.experiments
        # dico["49"]["scores"]["FROT_1.0"]["100"]["regular"]

        seeds = list(dico)
        methods = list(dico[seeds[0]]["scores"])

        up = False
        scores_means = {}
        for method in methods:
            score = [np.mean([dico[seed]["scores"][method][ngroup][balanced]
                              for seed in seeds]) for ngroup in groups]
            if np.min(score) < 0.8:
                up = True
                
            if frots:
                if not method.startswith("FROT"):
                    continue
            else:
                if method.startswith("FROT"):
                    eta = float(method.split("_")[1])
                    if not eta == 1.0:
                        continue
                    method = "FROT"
                        
            scores_means[method] = score

        plt.figure()
        
        # plt.title('Accuracy of a 2 class SVM trained on the selected features')
        plt.ylabel('Accuracy')
        plt.xlabel('Best k features')

        nmethod = len(scores_means)
        step = 0.4
        width = 2*step/(nmethod)

        ind = np.arange(len(groups))
        
        for index, ((methodname, model_scores), loc) in enumerate(zip(scores_means.items(), np.linspace(-step, step, nmethod+1)[1:])):
            plt.bar(ind+loc-width/2, model_scores, width=width, label='{}'.format(methodname))
            # import ipdb; ipdb.set_trace()
        
        plt.xticks(ind, groups)
        plt.yticks(np.linspace(0, 1, 11))
        plt.ylim(0.7, 1)

        if up:
            bbox_to_anchor = (0.98, 0.96)
            loc = 'upper right'
        else:
            bbox_to_anchor = (0.98, 0.04)
            loc = 'lower right'
        
        plt.legend(bbox_to_anchor=bbox_to_anchor,
                   borderaxespad=0.,
                   loc=loc,
                   frameon=True, framealpha=1, fancybox=True)
        plt.savefig(figname) # bbox_inches="tight")
        # plt.show()
        plt.close()

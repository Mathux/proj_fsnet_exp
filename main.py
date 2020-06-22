import click
from tqdm import tqdm
from src.dataloader import DataLoaders


@click.command()
@click.argument('dataset', type=click.Choice(DataLoaders.keys()))
def main(dataset):
    from src.tools import fit_and_time
    from src.models.hsic import HSICLasso
    from src.models.mRMR import mRMR
    from src.evaluator import Experiments
    
    dataLoader = DataLoaders[dataset]
    Algorithms = [HSICLasso, mRMR]
    # MMD/mRMR/HSCI Lasso
    
    # nfeats = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    nfeats = [10]
    nepochs = 2
    
    experiments = Experiments(dataset)
    pbar = tqdm([0] + list(range(nepochs))) # add 0 again to do it twice, to remove initialization time for python modules
    for seed in pbar:
        data = dataLoader(seed=seed)
        methods = [alg(k=k) for k in nfeats for alg in Algorithms]
        models = {model.modelname: model for model in methods}
        for modelname, model in models.items():
            pbar.set_description("Fitting {}".format(modelname))
            fit_and_time(model)(data.X_train, data.y_train)
            
        experiments.add_experiment(seed, data, models)
        
    experiments.save()


if __name__ == "__main__":
    main()

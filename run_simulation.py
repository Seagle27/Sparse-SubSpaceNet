from pathlib import Path
from datetime import datetime
import sys
import torch

from src.config.simulation_config import SimulationConfig
from src.system_model import SystemModel
from src.models import ModelGenerator
from src.signal_creation import Samples
from src.training import TrainingParams, train
from src.evaluation import evaluate
from src.data_handler import create_dataset, load_datasets, SameLengthBatchSampler, collate_fn
from src.plotting import initialize_figures, plot_results
from src.training import set_criterions
from src.utils import print_loss_results_from_simulation
from src.steering_vector_generator import SteeringVectorGenerator


class SimulationRunner:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.base_path = Path(__file__).parent / "data"
        self.paths = self._init_paths()
        self.monte_carlo_simulations = 1


    def _init_paths(self):
        paths = {
            "datasets": self.base_path / "datasets" / "uniform_bias_spacing",
            "saving": self.base_path / "weights",
            "simulations": self.base_path / "simulations"
        }
        for p in [paths["datasets"] / "train", paths["datasets"] / "test", paths["saving"] / "final_models"]:
            p.mkdir(parents=True, exist_ok=True)
        return paths

    def train_model(self, model_gen, train_dataset):
        config = self.config
        simulation_parameters = (
            TrainingParams()
            .set_training_objective(config.training.training_objective)
            .set_batch_size(config.training.batch_size)
            .set_epochs(config.training.epochs)
            .set_model(model_gen=model_gen)
            .set_optimizer(config.training.optimizer,
                           config.training.learning_rate,
                           config.training.weight_decay)
            .set_training_dataset(train_dataset)
            .set_schedular(config.training.scheduler,
                           config.training.step_size,
                           config.training.gamma)
            .set_criterion(config.training.criterion, config.training.balance_factor)
        )

        model, _, _ = train(
            training_parameters=simulation_parameters,
            saving_path=self.paths["saving"],
            save_figures=config.commands.save_plots,
            plot_curves=config.commands.plot_results
        )

        if config.commands.save_model:
            torch.save(model.state_dict(),
                       self.paths["saving"] / "final_models" / model.get_model_file_name())

        return model

    def evaluate_model(self, model, system_model, test_dataset):
        config = self.config
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            collate_fn=collate_fn,
            batch_sampler=SameLengthBatchSampler(test_dataset, batch_size=1),
            shuffle=False
        )

        criterion, _ = set_criterions(
            config.evaluation.criterion,
            config.evaluation.balance_factor
        )

        figures = initialize_figures()

        return evaluate(
            generic_test_dataset=test_loader,
            criterion=criterion,
            system_model=system_model,
            figures=figures,
            plot_spec=False,
            models=config.evaluation.models,
            augmented_methods=config.evaluation.augmented_methods,
            subspace_methods=config.evaluation.subspace_methods,
            model_tmp=model
        )

    def _run_single_simulation(self):
        config = self.config

        # Redirect stdout to file if enabled
        if config.commands.save_to_file:
            now = datetime.now().strftime("%d_%m_%Y_%H_%M")
            suffix = ""
            if config.commands.train_model:
                suffix += f"_train_{config.model.model_type}_{config.training.training_objective}"
            suffix += (f"_{config.system_model.signal_nature}_SNR_{config.system_model.snr}_T_{config.system_model.T}"\
                       f"_eta{config.system_model.eta}.txt")
            log_file = self.paths["simulations"] / "results" / "scores" / Path(now + suffix)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            self.orig_stdout = sys.stdout
            sys.stdout = open(log_file, "w")

        system_model = SystemModel(config.system_model)
        model_gen = (
            ModelGenerator()
            .set_model_type(config.model.model_type)
            .set_system_model(system_model)
            .set_model_params(config.model.model_params)
            .set_model()
        )
        samples_model = Samples(config.system_model, config.system_model.antenna_pattern)

        train_dataset = None
        test_dataset = None

        if config.commands.create_data:
            if config.commands.train_model:
                train_dataset = create_dataset(
                    samples_model, config.training.samples_size, config.commands.save_dataset,
                    self.paths["datasets"], config.training.true_doa_train,
                    config.training.true_range_train, phase="train"
                )

            if config.commands.evaluate_mode:
                test_dataset = create_dataset(
                    samples_model, int(config.training.train_test_ratio * config.training.samples_size), config.commands.save_dataset,
                    self.paths["datasets"], config.training.true_doa_test,
                    config.training.true_range_test, phase="test"
                )
        else:
            try:
                if config.commands.train_model:
                    train_dataset = load_datasets(
                        config.system_model,
                        config.training.samples_size,
                        self.paths["datasets"],
                        is_training=True
                    )
                if config.commands.evaluate_mode:
                    test_dataset = load_datasets(
                        config.system_model,
                        int(config.training.train_test_ratio * config.training.samples_size),
                        self.paths["datasets"],
                        is_training=False
                    )
            except Exception as e:
                print("Fallback to data creation due to error:", e)
                config.commands.create_data = True
                return self._run_single_simulation()

        model = None
        if config.commands.train_model:
            model = self.train_model(model_gen, train_dataset)

        result = None
        if config.commands.evaluate_mode:
            result = self.evaluate_model(model, system_model, test_dataset)

        if config.commands.save_to_file:
            sys.stdout.close()
            sys.stdout = self.orig_stdout
            
        SteeringVectorGenerator.reset_instance()

        return result

    def run(self):
        if not self.config.scenario:
            return self._run_single_simulation()

        loss_dict = {}
        for key, values in self.config.scenario.items():
            loss_dict[key] = {}
            for val in values:
                setattr(self.config.system_model, key, val)
                loss, successful_simulations = {}, 0
                for i in range(self.monte_carlo_simulations):
                    
                    print(f"Running scenario: {key} = {val}, simulation step = {i}")
                    result = self._run_single_simulation()
                    if result is not None:
                        successful_simulations += 1
                        if not loss:
                            loss = result
                        else:
                            for k in result.keys():
                                loss[k]['Overall'] += result[k]['Overall']
                for k in loss.keys():
                    loss[k]['Overall'] = loss[k]['Overall'] / successful_simulations
                loss_dict[key][val] = loss
                print(loss_dict)

        if None not in list(next(iter(loss_dict.values())).values()):
            print_loss_results_from_simulation(loss_dict)
            # if self.config.commands.plot_results:
            #     plot_results(
            #         loss_dict,
            #         criterion=self.config.evaluation.criterion,
            #         plot_acc=False,
            #         save_to_file=self.config.commands.save_plots,
            #     )

        return loss_dict

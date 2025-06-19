"""
Subspace-Net

Details
----------
Name: evaluation.py
Authors: D. H. Shmuel
Created: 01/10/21
Edited: 17/03/23

Purpose
----------
This module provides functions for evaluating the performance of Subspace-Net and others Deep learning benchmarks,
add for conventional subspace methods. 
This scripts also defines function for plotting the methods spectrums.
In addition, 


Functions:
----------
evaluate_dnn_model: Evaluate the DNN model on a given dataset.
evaluate_augmented_model: Evaluate an augmented model that combines a SubspaceNet model.
evaluate_model_based: Evaluate different model-based algorithms on a given dataset.
add_random_predictions: Add random predictions if the number of predictions
    is less than the number of sources.
evaluate: Wrapper function for model and algorithm evaluations.


"""
# Imports
import os
import time
import numpy as np
import torch.linalg
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from pathlib import Path

# Internal imports
from src.utils import *
from src.criterions import RMSPELoss, ADMMObjective
from src.methods import MVDR
from src.methods_pack.music import MUSIC
from src.methods_pack.root_music import RootMusic
from src.methods_pack.esprit import ESPRIT
from src.methods_pack.mle import MLE
from src.models import (ModelGenerator, SubspaceNet, DCDMUSIC, DeepAugmentedMUSIC,
                        DeepCNN, DeepRootMUSIC, TransMUSIC, SparseNet)
from src.plotting import plot_spectrum
from src.system_model import SystemModel
from src.config.simulation_config import SystemModelParams
from src.methods_pack.cov_reconstruct import CovReconstructor, get_cov_reconstruction_method, sample_covariance


def get_model_based_method(method_name: str, system_model: SystemModel):
    """

    Parameters
    ----------
    method_name(str): the method to use - music_1d, music_2d, root_music, esprit...
    system_model(SystemModel) : the system model to use as an argument to the method class.

    Returns
    -------
    an instance of the method.
    """
    if method_name.lower().endswith("music_1d"):
        return MUSIC(system_model=system_model, estimation_parameter="angle")
    if method_name.lower().endswith("2d-music"):
        return MUSIC(system_model=system_model, estimation_parameter="angle, range")
    if method_name.lower() == "root_music":
        return RootMusic(system_model)
    if method_name.lower().endswith("esprit"):
        return ESPRIT(system_model)


def get_model(model_name: str, params: dict, system_model: SystemModel):
    model_config = (
        ModelGenerator()
        .set_model_type(model_name)
        .set_system_model(system_model)
        .set_model_params(params)
        .set_model()
    )
    model = model_config.model
    path = os.path.join(Path(__file__).parent.parent, "data", "weights", "final_models", model.get_model_file_name())
    try:
        model.load_state_dict(torch.load(path))
    except FileNotFoundError as e:
        print("####################################")
        print(e)
        print("####################################")
        try:
            print(f"Model {model_name} not found in final_models, trying to load from temp weights.")
            path = os.path.join(Path(__file__).parent.parent, "data", "weights", model.get_model_file_name())
            model.load_state_dict(torch.load(path))
        except FileNotFoundError as e:
            print("####################################")
            print(e)
            print("####################################")
            warnings.warn(f"get_model: Model {model_name} not found")
    return model.to(device)


def evaluate_dnn_model(model: nn.Module, dataset: DataLoader, mode: str="valid") -> dict:
    """
    Evaluate the DNN model on a given dataset.

    Args:
        model (nn.Module): The trained model to evaluate.
        dataset (DataLoader): The evaluation dataset.

    Returns:
        float: The overall evaluation loss.

    Raises:
        Exception: If the evaluation loss is not implemented for the model type.
    """

    # Initialize values
    overall_loss_angle = 0.0
    overall_accuracy = 0.0
    test_length = 0
    # Set model to eval mode
    model.eval()
    with (torch.no_grad()):
        for data in dataset:
            if mode == "valid":
                eval_loss = model.validation_step(data)
            else:
                eval_loss = model.test_step(data)

            if isinstance(eval_loss, tuple):
                eval_loss, acc = eval_loss
            else:
                acc = None

            overall_loss_angle += torch.sum(eval_loss).item()
            if acc is not None:
                if overall_accuracy is None:
                    overall_accuracy = 0.0
                overall_accuracy += acc
            if data[0].dim() == 2:
                test_length += 1
            else:
                test_length += data[0].shape[0]
            ############################################################################################################
    overall_loss_angle /= test_length
    if overall_accuracy is not None:
        overall_accuracy /= test_length
    overall_loss = {"Angle": overall_loss_angle,
                    "Accuracy": overall_accuracy}

    return overall_loss


def evaluate_augmented_model(
        model: SubspaceNet,
        dataset,
        system_model,
        criterion=RMSPELoss,
        algorithm: str = "music",
        plot_spec: bool = False,
        figures: dict = None,
):
    """
    Evaluate an augmented model that combines a SubspaceNet model with another subspace method on a given dataset.

    Args:
    -----
        model (nn.Module): The trained SubspaceNet model.
        dataset: The evaluation dataset.
        system_model (SystemModel): The system model for the hybrid algorithm.
        criterion: The loss criterion for evaluation. Defaults to RMSPE.
        algorithm (str): The hybrid algorithm to use (e.g., "music", "mvdr", "esprit"). Defaults to "music".
        plot_spec (bool): Whether to plot the spectrum for the hybrid algorithm. Defaults to False.
        figures (dict): Dictionary containing figure objects for plotting. Defaults to None.

    Returns:
    --------
        float: The average evaluation loss.

    Raises:
    -------
        Exception: If the algorithm is not supported.
        Exception: If the algorithm is not supported
    """
    # Initialize parameters for evaluation
    hybrid_loss = []
    if not isinstance(model, SubspaceNet):
        raise Exception("evaluate_augmented_model: model is not from type SubspaceNet")
    # Set model to eval mode
    model.eval()
    # Initialize instances of subspace methods
    methods = {
        "mvdr": MVDR(system_model),
        "music": MUSIC(system_model, estimation_parameter="angle"),
        "esprit": ESPRIT(system_model),
        "r-music": RootMusic(system_model),
        "music_2D": MUSIC(system_model, estimation_parameter="angle, range")
    }
    # If algorithm is not in methods
    if methods.get(algorithm) is None:
        raise Exception(
            f"evaluate_augmented_model: Algorithm {algorithm} is not supported."
        )
    # Gradients calculation isn't required for evaluation
    with torch.no_grad():
        for i, data in enumerate(dataset):
            X, true_label = data
            if algorithm.endswith("2D"):
                DOA, RANGE = torch.split(true_label, true_label.size(1) // 2, dim=1)
                RANGE.to(device)
            else:
                DOA = true_label

            # Convert observations and DoA to device
            X = X.to(device)
            DOA = DOA.to(device)
            # Apply method with SubspaceNet augmentation
            method_output = methods[algorithm].narrowband(
                X=X, mode="SubspaceNet", model=model
            )
            # Calculate loss, if algorithm is "music" or "esprit"
            if not algorithm.startswith("mvdr"):
                if algorithm.endswith("2D"):
                    predictions_doa, predictions_distance = method_output[0], method_output[1]
                    loss = criterion(predictions_doa, DOA * R2D, predictions_distance, RANGE)
                else:
                    predictions, M = method_output[0], method_output[-1]
                    # If the amount of predictions is less than the amount of sources
                    predictions = add_random_predictions(M, predictions, algorithm)
                    # Calculate loss criterion
                    loss = criterion(predictions, DOA * R2D)
                hybrid_loss.append(loss)
            else:
                hybrid_loss.append(0)
            # Plot spectrum, if algorithm is "music" or "mvdr"
            if not algorithm.startswith("esprit"):
                if plot_spec and i == len(dataset.dataset) - 1:
                    predictions, spectrum = method_output[0], method_output[1]
                    figures[algorithm]["norm factor"] = np.max(spectrum)
                    plot_spectrum(
                        predictions=predictions,
                        true_DOA=DOA * R2D,
                        system_model=system_model,
                        spectrum=spectrum,
                        algorithm="SubNet+" + algorithm.upper(),
                        figures=figures,
                    )
    return np.mean(hybrid_loss)


def evaluate_model_based(
        dataset: DataLoader,
        system_model: SystemModel,
        criterion: nn.Module,
        algorithm: str,
        cov_recon: CovReconstructor):
    """
    Evaluate different model-based algorithms on a given dataset.

    Args:
        dataset (DataLoader): The evaluation dataset.
        system_model (SystemModel): The system model for the algorithms.
        criterion (nn.Module): The loss criterion for evaluation. Defaults to RMSPE.
        algorithm (str): The algorithm to use (e.g., "music", "mvdr", "esprit", "r-music"). Defaults to "music".
        cov_recon (CovReconstructor) : The method to use for the covariance matrix reconstruction

    Returns:
        overall_loss (Dict): Dict of evaluation loss and accuracy

    Raises:
        Exception: If the algorithm is not supported.
    """
    # Initialize parameters for evaluation
    overall_loss = 0.0
    overall_acc = 0.0
    test_length = 0

    model_based = get_model_based_method(algorithm, system_model)
    if isinstance(model_based, nn.Module):
        model_based = model_based.to(device)
        # Set model to eval mode
        model_based.eval()
    # Gradients calculation isn't required for evaluation
    with torch.no_grad():
        for i, data in enumerate(dataset):
            x, sources_num, angles = data
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = x.to(device)
            angles = angles.to(device)
            validate_constant_sources_number(sources_num)
            sources_num = sources_num[0]

            cov = cov_recon(x)
            angles_prediction, source_estimation, _ = model_based(cov, sources_num=sources_num)
            overall_loss += criterion(angles_prediction, angles).item()
            overall_acc += torch.sum(source_estimation == sources_num * torch.ones_like(source_estimation).float()).item()

            if data[0].dim() == 2:
                test_length += 1
            else:
                test_length += data[0].shape[0]

        overall_loss /= test_length
        overall_acc /= test_length
        overall_loss = {"Angle": overall_loss,
                        "Accuracy": overall_acc}
        return overall_loss


def evaluate_admm_convergence(dataset: DataLoader,
                              criterion: nn.Module,
                              cov_recon: CovReconstructor):
    # Initialize parameters for evaluation
    overall_loss = 0.0
    test_length = 0

    # Gradients calculation isn't required for evaluation
    with torch.no_grad():
        for i, data in enumerate(dataset):
            x, sources_num, _ = data
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = x.to(device)
            validate_constant_sources_number(sources_num)

            cov = cov_recon(x)
            Rx = sample_covariance(x)
            overall_loss += criterion(cov, Rx)
            if data[0].dim() == 2:
                test_length += 1
            else:
                test_length += data[0].shape[0]

        overall_loss /= test_length
        return overall_loss


def add_random_predictions(M: int, predictions: np.ndarray, algorithm: str):
    """
    Add random predictions if the number of predictions is less than the number of sources.

    Args:
        M (int): The number of sources.
        predictions (np.ndarray): The predicted DOA values.
        algorithm (str): The algorithm used.

    Returns:
        np.ndarray: The updated predictions with random values.

    """
    # Convert to np.ndarray array
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    while predictions.shape[0] < M:
        # print(f"{algorithm}: cant estimate M sources")
        predictions = np.insert(
            predictions, 0, np.round(np.random.rand(1) * 180, decimals=2) - 90.00
        )
    return predictions


def evaluate_crb(dataset: list,
                 params: SystemModelParams,
                 mode: str="separate"):
    u_snr = 10 ** (params.snr / 10)
    if params.field_type.lower() == "far":
        print("CRB calculation is not supported for Far Field yet.")
        return None
    elif params.field_type.lower() == "near":
        if params.signal_nature.lower() == "non-coherent":
            angles = []
            distances = []
            ucrb_cartzien = None
            for i, data in enumerate(dataset):
                _, _, labels, _ = data
                angles.extend(*labels[:, :labels.shape[1] // 2][None, :].detach().numpy())
                distances.extend(*labels[:, labels.shape[1] // 2:][None, :].detach().numpy())
            angles = np.array(angles)
            distances = np.array(distances)
            snr_coeff = (1 + 1 / (u_snr * params.N))
            ucrb_angle = (3 * 2 ** 2) / (2 * u_snr * params.T * (np.pi * np.cos(angles)) ** 2)
            ucrb_angle *= (8 * params.N - 11) * (2 * params.N - 1)
            ucrb_angle /= params.N * (params.N ** 2 - 1) * (params.N ** 2 - 4)
            ucrb_angle *= snr_coeff

            ucrb_distance = 6 * distances ** 2 * 2 ** 4 / (u_snr * params.T * np.pi ** 2)  # missing /wavelength
            ucrb_distance *= snr_coeff
            ucrb_distance /= params.N ** 2 * (params.N ** 2 - 1) * (params.N ** 2 - 4) * np.cos(angles) ** 4
            num = 15 * distances ** 2
            num += (30 / 2) * distances * (params.N - 1) * np.sin(angles)  # missing *wavelength
            num += (1 / 2) ** 2 * (8 * params.N - 11) * (2 * params.N - 1) * np.sin(angles) ** 2  # missing * wavelength ** 2
            ucrb_distance *= num
            if mode == "cartesian":
                # Need to calculate the cross term as well, and change coordinates.
                ucrb_cross = - snr_coeff * (3 * distances)
                ucrb_cross /= u_snr * params.T * np.pi ** 2 * (1 / 2) ** 3
                ucrb_cross *= 15 * distances*(params.N - 1) + (1 / 2) * (8 * params.N - 11) * (2 * params.N - 1) * np.sin(angles)
                ucrb_cross /= params.N * (params.N ** 2 - 1) * (params.N ** 2 - 4) * np.cos(angles) ** 3

                #change coordinates
                ucrb_cartzien = distances ** 2 * ucrb_angle + ucrb_distance
                ucrb_cartzien -= distances ** 2 * np.sin(2 * angles) * ucrb_angle
                # ucrb_cartzien += np.sin(2 * angles) * ucrb_distance
                # ucrb_cartzien += 2 * distances * np.cos(2 * angles) * ucrb_cross
                ucrb_cartzien = np.mean(ucrb_cartzien)

            return {"Overall": ucrb_cartzien, "Angle": np.mean(ucrb_angle), "Distance": np.mean(ucrb_distance)}
        else:
            print("UCRB calculation for the coherent is not supported yet")
    else:
        print("Unrecognized field type.")
    return


def evaluate_mle(dataset: list, system_model: SystemModel, criterion):
    """
    Evaluate the Maximum Likelihood Estimation (MLE) algorithm on a given dataset.

    Args:
        dataset (list): The evaluation dataset.
        system_model (SystemModel): The system model for the MLE algorithm.

    Returns:
        float: The average evaluation loss.
    """
    # initialize mle instance
    mle = MLE(system_model)
    # Initialize parameters for evaluation
    loss_list = []
    for i, data in enumerate(dataset):
        X, labels = data
        Rx = calculate_covariance_tensor(X, method="simple").to(device)
        angles = labels[:, :labels.shape[-1] // 2].to(device)
        distances = labels[:, labels.shape[-1] // 2:].to(device)
        # Apply MLE algorithm
        pred_angle, pred_distance = mle(Rx)
        # Calculate loss criterion
        loss = criterion(pred_angle.to(device), angles, pred_distance.to(device), distances)
        loss_list.append(loss.item())
    return {"Overall": np.mean(loss_list)}


def evaluate(
        generic_test_dataset: DataLoader,
        criterion: nn.Module,
        system_model: SystemModel,
        models: dict = None,
        augmented_methods: list = None,
        subspace_methods: list = None,
        model_tmp: nn.Module = None,
        cov_recon_method = 'sample',
        cov_recon_params: dict = None):
    """
    TODO: Update docs
    Wrapper function for model and algorithm evaluations.

    Parameters:
        generic_test_dataset (list): Test dataset for generic subspace methods.
        criterion (nn.Module): Loss criterion for (DNN) model evaluation.
        system_model: instance of SystemModel.
        figures (dict): Dictionary to store figures.
        plot_spec (bool, optional): Whether to plot spectrums. Defaults to True.
        models (dict): dict that contains the models to evluate and their parameters.
        augmented_methods (list, optional): List of augmented methods for evaluation.
            Defaults to None.
        subspace_methods (list, optional): List of subspace methods for evaluation.
            Defaults to None.

    Returns:
        None
    """
    res = {}
    # Evaluate DNN model if given
    if model_tmp is not None:
        model_test_loss = evaluate_dnn_model(model_tmp, generic_test_dataset)
        try:
            model_name = model_tmp._get_name()
        except AttributeError:
            model_name = "DNN"
        res[model_name + "_tmp"] = model_test_loss
    # Evaluate DNN models
    for model_name, params in models.items():
        model = get_model(model_name, params, system_model)
        # num_of_params = sum(p.numel() for p in model.parameters())
        # total_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
        # print(f"Number of parameters in {model_name}: {num_of_params} with total size: {total_size} bytes")
        start = time.time()
        model_test_loss = evaluate_dnn_model(model_tmp, generic_test_dataset)
        print(f"{model_name} evaluation time: {time.time() - start}")
        res[model_name] = model_test_loss

    # Evaluate SubspaceNet augmented methods  #TODO: FIX and cleanup augmented methods
    # for algorithm in augmented_methods:
    #     loss = evaluate_augmented_model(
    #         model=model,
    #         dataset=generic_test_dataset,
    #         system_model=system_model,
    #         criterion=criterion,
    #         algorithm=algorithm,
    #         plot_spec=plot_spec,
    #         figures=figures,
    #     )
    #     res["augmented" + algorithm] = loss

    # Evaluate classical subspace methods
    cov_recon = get_cov_reconstruction_method(cov_recon_method, system_model, **cov_recon_params)
    if isinstance(criterion, ADMMObjective):
        loss = evaluate_admm_convergence(generic_test_dataset, criterion, cov_recon)
        res[cov_recon.__class__.__name__] = loss
    else:
        for algorithm in subspace_methods:
            start = time.time()
            loss = evaluate_model_based(
                generic_test_dataset,
                system_model,
                criterion=criterion,
                algorithm=algorithm,
                cov_recon=cov_recon)
            if system_model.params.signal_nature == "coherent" and algorithm.lower() in ["1d-music", "2d-music", "r-music", "esprit"]:
                algorithm += "(SPS)"
            print(f"{algorithm} evaluation time: {time.time() - start}")
            res[algorithm] = loss

    for method, loss_ in res.items():
        print(f"{method.upper() + ' test loss' : <30} = {loss_}")
    return res

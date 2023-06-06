"""Subspace-Net main script 
    Details
    -------
    Name: main.py
    Authors: D. H. Shmuel
    Created: 01/10/21
    Edited: 17/03/23

    Purpose
    --------
    This script allows the user to apply the proposed algorithms,
    by wrapping all the required procedures and parameters for the simulation.
    This scripts calls the following functions:
        * create_dataset: For creating training and testing datasets 
        * run_simulation: For training DR-MUSIC model
        * evaluate_model: For evaluating subspace hybrid models

    This script requires that requirements.txt will be installed within the Python
    environment you are running this script in.

"""
###############
#   Imports   #
###############

import sys
import torch
import os
import matplotlib.pyplot as plt
import warnings
from src.system_model import *
from src.signal_creation import *
from src.data_handler import *
from src.criterions import *
from src.methods import *
from src.models import *
from src.simulation_handler import *
from src.utils import * 

warnings.simplefilter("ignore")
os.system('cls||clear')
plt.close('all')

if __name__ == "__main__":
    # Set relevant paths
    Main_path           = r"C:\Users\dorsh\Deep RootMUSIC\Code"
    Main_Data_path      = Main_path + "\DataSet"
    saving_path         = Main_path + "\Weights"
    simulations_path    = Main_path + "\Simulations"
    # Data_Scenario_path  = r"\CoherentSourcesWithGap"
    Data_Scenario_path  = r"\LowSNR"
    
    # Initialize time and date
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")

    # Define compilation device and set seed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_unified_seed()
    
    ## Operation commands
    commands = {"SAVE_TO_FILE"  : False,     # Saving results to file or present them over CMD
                "CREATE_DATA"   : True,     # Creating new data
                "LOAD_DATA"     : False,    # Loading data from dataset 
                "TRAIN_MODE"    : True,     # Applying training operation
                "SAVE_MODEL"    : False,    # Saving tuned model
                "EVALUATE_MODE" : True}     # Evaluating desired algorithms
                
    if commands["SAVE_TO_FILE"]:
        file_path = simulations_path + r"\\Results\\Scores\\" + dt_string_for_save + r".txt"
        sys.stdout = open(file_path, "w")

    print("------------------------------------")
    print("---------- New Simulation ----------")
    print("------------------------------------")
    print("date and time =", dt_string)
    
    # Set model type, options: "DA-MUSIC", "DeepCNN" 
    model_type = "SubspaceNet" 
    # Number of lags, relevant for "SubspaceNet" model 
    tau = 8     
    ## System model parameters
    N = 8       # Number of sensors
    M = 2       # number of sources
    T = 200     # Number of observations, ideal >= 200
    SNR = 10    # Signal to noise ratio, ideal = 10
    # Signal parameters
    scenario = "NarrowBand" # signals type, options: "NarrowBand", "Broadband_OFDM", "Broadband_simple"
    mode = "non-coherent"   # signals nature, options: "non-coherent", "coherent"
    # Array calibration values
    eta = 0                     # Deviation from sensor location, normalized by wavelength, ideal = 0
    geo_noise_var = 0           # Added noise for sensors response
    # simulation parameters
    samples_size = 10       # Overall dateset size
    train_test_ratio = 0.1     # training and testing datasets ratio 

    ############################
    #     Create Data Sets     #  
    ############################
    
    if commands["CREATE_DATA"]:
        set_unified_seed()
        Create_Training_Data = True # Flag for creating training data
        Create_Testing_Data = True  # Flag for creating test data
        
        print("Creating Data...")
        if Create_Training_Data:
            ## Training Datasets
            train_dataset, _, _ = create_dataset(
                                    scenario= scenario,
                                    mode= mode,
                                    N= N, M= M , T= T,
                                    samples_size = samples_size,
                                    tau = tau,
                                    model_type = model_type,
                                    Save = False,
                                    dataset_path = Main_Data_path + Data_Scenario_path + r"\TrainingData",
                                    true_doa = None,
                                    SNR = SNR,
                                    eta=eta,
                                    geo_noise_var = geo_noise_var,
                                    phase = "train")
        if Create_Testing_Data:
            ## Test Datasets
            test_dataset, generic_test_dataset, Sys_Model = create_dataset(
                                    scenario = scenario,
                                    mode = mode,
                                    N= N, M= M , T= T,
                                    samples_size = int(train_test_ratio * samples_size),
                                    tau = tau,
                                    model_type = model_type,
                                    Save = False,
                                    dataset_path= Main_Data_path + Data_Scenario_path + r"\TestData",
                                    true_doa =  None,
                                    SNR = SNR,
                                    eta = eta,
                                    geo_noise_var = geo_noise_var,
                                    phase = "test")

    ############################
    ###    Load Data Sets    ###
    ############################
    
    if commands["LOAD_DATA"]:
        try:
            TRAIN_DATA_PATH = '_{}_{}_{}_M={}_N={}_T={}_SNR={}_eta={}_geo_noise_var{}.h5'.format(scenario, mode, samples_size, M, N,
                                    T, SNR, str(eta).replace(",", ""), str(geo_noise_var).replace(",", ""))
            TEST_DATA_PATH = '_{}_{}_{}_M={}_N={}_T={}_SNR={}_eta={}_geo_noise_var{}.h5'.format(scenario, mode, int(train_test_ratio * samples_size),
                                    M, N, T, SNR, str(eta).replace(",", ""), str(geo_noise_var).replace(",", ""))

            train_dataset = read_data(Main_Data_path + Data_Scenario_path + f"\\TrainingData\\{model_type}_DataSet" + TRAIN_DATA_PATH)
            test_dataset  = read_data(Main_Data_path + Data_Scenario_path + f"\\TestData\\{model_type}_DataSet"     + TEST_DATA_PATH)
            Sys_Model     = read_data(Main_Data_path + Data_Scenario_path + r"\\TestData\\Sys_Model"      + TEST_DATA_PATH)
        
        except:
            TRAIN_DATA_PATH = '_{}_{}_{}_M={}_N={}_T={}_SNR={}_eta={}_geo_noise_var{}.h5'.format(scenario, mode, samples_size, M, N,
                                    T, SNR, str(eta).replace(",", ""), str(geo_noise_var).replace(",", ""))
            TEST_DATA_PATH = '_{}_{}_{}_M={}_N={}_T={}_SNR={}_eta={}_geo_noise_var{}.h5'.format(scenario, mode, int(train_test_ratio * samples_size),
                                    M, N, T, SNR, str(eta).replace(",", ""), str(geo_noise_var).replace(",", ""))

            DataSet_Rx_train = read_data(Main_Data_path + Data_Scenario_path + r"\\TrainingData\\DataSet_Rx" + TRAIN_DATA_PATH)
            DataSet_x_train  = read_data(Main_Data_path + Data_Scenario_path + r"\\TrainingData\\DataSet_x"   + TRAIN_DATA_PATH)
            DataSet_Rx_test  = read_data(Main_Data_path + Data_Scenario_path + r"\\TestData\\DataSet_Rx"     + TEST_DATA_PATH)
            DataSet_x_test   = read_data(Main_Data_path + Data_Scenario_path + r"\\TestData\\DataSet_x"      + TEST_DATA_PATH)
            Sys_Model        = read_data(Main_Data_path + Data_Scenario_path + r"\\TestData\\Sys_Model"      + TEST_DATA_PATH)

    ############################
    ###    Training stage    ###
    ############################
    
    if commands["TRAIN_MODE"]:
        # Training aided parameters
        optimal_lr = 0.00001         # Learning rate value
        optimal_bs = 256            # Batch size value
        epochs = 50                # Number of epochs
        optimal_step = 100          # Number of steps for learning rate decay iteration
        optimal_gamma_val = 0.9      # learning rate decay value
        weight_decay_val = 1e-9

        # list containers declaration
        Test_losses = []
        train_loss_lists = []
        validation_loss_lists = []
        train_curves = []
        validation_curves = []

        ############################
        ###    Run Simulation    ###
        ############################

        print("\n--- New Simulation ---\n")
        # print("Description: Simulation of broadband sources within range [0-500] Hz with T = {}, Tau = {}, SNR = {}, {} sources".format(T, tau, SNR, mode))
        # print("Description: Simulation with constant {} deviation in sensors location, T = {}, SNR = {}, {} sources".format(eta, T, SNR, mode))
        print("Description: Simulation of multiple coherent sources", end=" ")
        print(f"Model: {model_type}")
        print(f"variance = {geo_noise_var} , T = {T}, SNR = {SNR}, {mode} sources")
        print("Simulation parameters:")
        print(f"Number of sensors = {N}")
        print(f"Number of sources = {M}")
        print(f"Learning Rate = {optimal_lr}")
        print(f"Weight decay = {weight_decay_val}")
        print("Batch Size = {}".format(optimal_bs))
        print("SNR = {}".format(SNR))
        print("scenario = {}".format(scenario))
        print("mode = {}".format(mode))
        print("Gamma Value = {}".format(optimal_gamma_val))
        print("Step Value = {}".format(optimal_step))
        print("Epochs = {}".format(epochs))
        print("Tau = {}".format(tau))
        print("Observations = {}".format(T))
        print("Spacing deviation (eta) = {}".format(eta))
        print("Geometry noise variance = {}".format(geo_noise_var))
            
        model, loss_train_list, loss_valid_list, Test_loss = run_simulation(
                        train_dataset = train_dataset,
                        test_dataset = test_dataset,
                        tau = tau,
                        optimizer_name = "Adam",
                        lr_val = optimal_lr,
                        schedular = True,
                        weight_decay_val = weight_decay_val,
                        step_size_val = optimal_step,
                        gamma_val = optimal_gamma_val,
                        num_epochs = epochs,
                        model_name= "{}_M={}_T={}_SNR_{}_tau={}_{}_{}_eta=0.0125".format(model_type, M, T, SNR, tau, scenario, mode, eta),
                        batch_size = optimal_bs,
                        system_model = Sys_Model,
                        load_flag = False,
                        loading_path = saving_path + r"\Models" + f"\{model_type}_M={M}_T={T}_SNR_{SNR}_tau={tau}_{scenario}",
                        Plot = False,
                        model_type = model_type)
        
        # Save model weights
        if commands["SAVE_MODEL"]:
            torch.save(model.state_dict(), saving_path + r"/Final_models" \
                    + r"/model_M={}_{}_Tau={}_SNR={}_T={}_eta={}".format(M, mode, tau, SNR, T, eta, geo_noise_var))
        
        train_loss_lists.append(loss_train_list)
        validation_loss_lists.append(loss_valid_list)
        Test_losses.append(Test_loss)
        
        # Plotting train & validation curves
        fig = plt.figure(figsize=(8, 6), dpi=80)
        plt.plot(range(epochs), loss_train_list, label="tr {}".format(optimal_lr))
        plt.plot(range(epochs), loss_valid_list, label="vl {}".format(optimal_lr))
        plt.title("Learning Curves: Loss per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(bbox_to_anchor=(0.95,0.5), loc="center left", borderaxespad=0)
        
        # Plots saving
        if commands["SAVE_TO_FILE"]:
            plt.savefig(simulations_path + r"\\Results\\Plots\\" + dt_string_for_save + r".png")
        else:
            plt.show()
    

    ############################
    ###   Evaluation stage   ###
    ############################
    
    if commands["EVALUATE_MODE"]:
        # augmented_methods = ["music", "mvdr", "esprit"]
        augmented_methods = []
        subspace_methods = []
        # subspace_methods = ["esprit", "music", "r-music",
        #                     "sps-r-music", "sps-esprit", "sps-music"]
                            # "bb-music"]
        # RootMUSIC_loss = []
        MUSIC_loss = []
        # SPS_RootMUSIC_loss = []
        SPS_MUSIC_loss = []
        DeepRootTest_loss = []
        if not (commands["CREATE_DATA"] or commands["LOAD_DATA"]):
            try:
                TEST_DATA_PATH = '_{}_{}_{}_M={}_N={}_T={}_SNR={}_eta={}_geo_noise_var{}.h5'.format(scenario, mode, int(train_test_ratio * samples_size),
                                        M, N, T, SNR, str(eta).replace(",", ""), str(geo_noise_var).replace(",", ""))
                model_test_dataset = read_data(Main_Data_path + Data_Scenario_path + f"\\TestData\\{model_type}_DataSet"     + TEST_DATA_PATH)
                generic_test_dataset = read_data(Main_Data_path + Data_Scenario_path + f"\\TestData\\Generic_DataSet"     + TEST_DATA_PATH)
                Sys_Model = read_data(Main_Data_path + Data_Scenario_path + r"\\TestData\\Sys_Model"      + TEST_DATA_PATH)

            except:
                TEST_DATA_PATH = '_{}_{}_{}_M={}_N={}_T={}_SNR={}_eta={}_geo_noise_var{}.h5'.format(scenario, mode, int(train_test_ratio * samples_size),
                                        M, N, T, SNR, str(eta).replace(",", ""), str(geo_noise_var).replace(",", ""))

                DataSet_Rx_train = read_data(Main_Data_path + Data_Scenario_path + r"\\TrainingData\\DataSet_Rx" + TRAIN_DATA_PATH)
                DataSet_x_train  = read_data(Main_Data_path + Data_Scenario_path + r"\\TrainingData\\DataSet_x"   + TRAIN_DATA_PATH)
                DataSet_Rx_test  = read_data(Main_Data_path + Data_Scenario_path + r"\\TestData\\DataSet_Rx"     + TEST_DATA_PATH)
                DataSet_x_test   = read_data(Main_Data_path + Data_Scenario_path + r"\\TestData\\DataSet_x"      + TEST_DATA_PATH)
                Sys_Model        = read_data(Main_Data_path + Data_Scenario_path + r"\\TestData\\Sys_Model"      + TEST_DATA_PATH)

        loss_measure = "rmse"
        print(f"Number of sensors = {N}")
        print(f"Number of sources = {M}")
        print("SNR = {}".format(SNR))
        print("scenario = {}".format(scenario))
        print("mode = {}".format(mode))
        print("Observations = {}".format(T))
        print("SNR = {}".format(SNR))
        print("Tau = {}".format(tau))
        print("Observations = {}".format(T))
        print("Spacing deviation (eta) = {}".format(eta))
        print("Geometry noise variance = {}".format(geo_noise_var))
        print(f"Loss measure = {loss_measure}")

        ############################
        ###    Load Data Set     ###
        ############################
        
        if not commands["TRAIN_MODE"]:
            # Load trained model
            model_path = r"\Models" + r"\model_30_05_2023_00_09" 
            # model_path = r"\Models" + f"\{model_type}_M=3_T={T}_SNR_{SNR}_tau={tau}_{scenario}_{mode}"
            # model_path = r"\Final_models" + r"\BroadBand" + r"/model_M={}_{}_Tau={}_SNR={}_T={}".format(M, mode, tau, SNR, T)
            loading_path = saving_path + model_path

            if model_type.startswith("DeepCNN"):
                model = DeepCNN(N, 361)
            elif model_type.startswith("SubspaceNet"):
                model = SubspaceNet(tau=tau, M=M)                                         
            elif model_type.startswith("DA-MUSIC"):
                model = DeepAugmentedMUSIC(N=N, T=T, M=M)
                    
            # Load the model to the specified device, either gpu or cpu
            model = model.to(device)
            try:
                if torch.cuda.is_available() is False:
                    model.load_state_dict(torch.load(loading_path, map_location=torch.device('cpu')))
            except:
                print("No loaded weights found")
                pass
            
        Data_Set_path = Main_path + r"\\DataSet"
        model_test_dataset = torch.utils.data.DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                drop_last=False)
        generic_test_dataset = torch.utils.data.DataLoader(generic_test_dataset,
                                batch_size=1,
                                shuffle=False,
                                drop_last=False)

        mb = ModelBasedMethods(Sys_Model)
        # Define loss measure for evaluation
        if loss_measure.startswith("rmse"):
            criterion = RMSPELoss() # define loss criterion
            hybrid_criterion = RMSPE
        elif loss_measure.startswith("mse"):
            criterion = MSPELoss() # define loss criterion
            hybrid_criterion = MSPE
            
        # Evaluate SubspaceNet augmented methods
        PLOT_SPECTRUM = False
        figures = {"music"  : {"fig" : None, "ax" : None, "norm factor" : None},
                "r-music": {"fig" : None, "ax" : None},
                "esprit" : {"fig" : None, "ax" : None},
                "mvdr"   : {"fig" : None, "ax" : None, "norm factor" : None}}

        model_test_loss = evaluate_model(model, model_test_dataset, criterion=criterion,
                                        plot_spec= PLOT_SPECTRUM, figures=figures, model_type = model_type)
        print(f"{model_type} Test loss = {model_test_loss}")
        for algorithm in augmented_methods:
            hybrid_loss = evaluate_hybrid_model(model, model_test_dataset, Sys_Model,
                            criterion = hybrid_criterion, algorithm = algorithm, plot_spec= PLOT_SPECTRUM, figures = figures)
            print("hybrid {} test loss = {}".format(algorithm, hybrid_loss))
        for algorithm in subspace_methods:
            loss = evaluate_model_based(generic_test_dataset, Sys_Model, criterion=hybrid_criterion,
                                        plot_spec = PLOT_SPECTRUM, algorithm = algorithm, figures = figures)
            print("{} test loss = {}".format(algorithm.lower(), loss))
        # figures["mvdr"]["fig"].savefig(f"mvdr_spectrum_M_{M}_{mode}_T_{T}_SNR_{SNR}.pdf", bbox_inches='tight')
        # figures["music"]["fig"].savefig("{}_spectrum.pdf".format("music"), bbox_inches='tight')
    plt.show()
    print("end")
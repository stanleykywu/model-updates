import os
import glob
import numpy as np
import random
from random import sample
from scipy.stats import norm
import torch
import tensorflow as tf
from tqdm import tqdm
        
class CarliniAttackUtilityKeras():
    
    def __init__(self, model, data_x, data_y, num_shadow_models=256, save=False, use_saved_models=False, saved_models_dir='experiments/CarliniShadowModels', model_name=None, seed=0):
        """Utility for performing membership inference attacks on deep 
        learning models in Keras using Carlini's 2021 attack from 
        first principles
        
            ...
            
            Attributes
            ----------
            
            model : Keras model
                The target model for the membership inference attack
            num_shadow_models : int
                The number of shadow models to use in the attack
            save : bool
                Flag to save shadow models after training
            save_dir : str
                If save is True, shadow models will save to experiments/CarliniShadowModels/<save_dir_prefix>/shadow_model_i.pth
            use_saved_models : bool
                Flag to use models from a directory instead of training from scratch
            saved_models_dir : str
                If use_saved_models is True, this is the directory the models will be pulled from
        """
        random.seed(seed)
        self.use_saved_models = use_saved_models
        self.__saved_models_dir = saved_models_dir
        self.model = model
        self.data_x = data_x
        self.data_y = data_y
        self.shadow_model = None
        self.__save_dir = model_name if model_name else str(type(model)).split(".")[-1].split("'")[0]
        self.save = save
        self.sgd = True if model_name == "Purchase" or model_name == "FMNIST" else False

        if not os.path.exists(self.__saved_models_dir): 
            os.mkdir(self.__saved_models_dir)
        
        if not self.use_saved_models:
            self.num_shadow_models = num_shadow_models
            self.shadow_model = tf.keras.models.clone_model(
                model, input_tensors=None, clone_function=None
            )

            if self.save:
                if not os.path.exists(f"{self.__saved_models_dir}/{self.__save_dir}"): os.mkdir(f"{self.__saved_models_dir}/{self.__save_dir}")
   
    def logit_scaling(self, p):
        """Perform logit scaling so that the model's confidence is 
        approximately normally distributed
            
            ...
            
            Parameters
            ----------
                p : numpy.ndarray
                    A list containing tensors of model's confidence scores
                    
            Returns
            -------
                phi(p) : numpy.ndarray
                    A list of scaled model confidence
        """
        return np.array([np.log(x/(1-x)) for x in p])
    
    
    def train_shadow_model(self, x, y, batch_size, lr=1e-4, epochs=5, shadow_model_number=0, adam=True):
        """Helper function to train individual shadow models

            ...

            Parameters
            ----------
                x : np.ndarray
                    x points training
                y : np.ndarray
                    y points training
                lr : float
                    Learning rate for shadow model optimizers
                epochs : int
                    Training epochs for shadow models
                shadow_model_number : int
                    Keeps track of how many shadow models have been trained for saving purposes
        """
        if adam:
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            opt = tf.keras.optimizers.SGD(learning_rate=lr)

        self.shadow_model.compile(
            loss=self.model.loss,
            optimizer=opt,
            metrics=["accuracy"],
        )
        self.shadow_model.fit(
            x,
            y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
        )
        score = self.shadow_model.evaluate(x, y, verbose=0)
        print(f'Shadow model {shadow_model_number} trained with acc: {score[1]}')
        
        if self.save and not self.use_saved_models:
            self.shadow_model.save(f"{self.__saved_models_dir}/{self.__save_dir}/shadow_model_{shadow_model_number}")

    def __sampling_helper(self, x, y, num_samples):
        """Take sample from entire data distribution

            ...

            Parameters
            ----------
                x: np.ndarray
                    Dataset that contains the entire data distribution for x
                y: np.ndarray
                    Dataset that contains the entire data distribution for y
                num_samples : int
                    The desired number of samples to train shadow models
                     
            Returns
            -------
                D_attack : np.ndarray
                    np.ndarray to train shadow model that contains a sample 
                    from the data distribution
                track_indices : NumPy array
                    Array of bools to keep track of which points were in the 
                    shadow model's training set
        """
        
        # Keeps track of which datapoints the shadow models trained on
        track_indices = np.zeros(len(x), dtype=bool)
        D_attack_indices = sample(range(len(x)), num_samples)
        track_indices[D_attack_indices] = True
        
        attack_x = x[D_attack_indices]
        attack_y = y[D_attack_indices]
        
        return (attack_x, attack_y)
        
    def model_confidence(self, model, x, y):
        """Helper function to calculate the model confidence on a dataset x and y

            Parameters
            ----------
                model : Keras model
                    A Keras machine learning model 
                x : np.ndarray
                    List of data for the example we want to perform membership inference on
                y : int
                    The labels associated to x
                    
            Returns
            -------
                model_confidence : np.ndarray
                    exp(-CrossEntropyLoss(x, y)) which is in [0, 1]
        """
        loss = torch.nn.CrossEntropyLoss(reduction="none", reduce=False)

        predictions = torch.tensor(model.predict(x))
        labels = y

        losses = loss(predictions, torch.tensor(labels)).numpy()
        return np.array([np.exp(-val) for val in losses])
    
    def offline_attack(self, attack_ds, batch_size, num_samples=5000, training_epochs=5, lr=1e-4):
        """Carlini's offline membership inference attack
        
            ...
            
            Parameters
            ----------
                attack_ds: tuple
                    Data x and y to run attack on
                batch_size : int
                    Batch size for shadow model training
                num_samples : int
                    The number of training points for each shadow model
                training_epochs : int
                    Number of epochs to train each shadow model
                lr : float
                    Learning rate for shadow models
        """
        confs_out = [[] for _ in range(len(attack_ds[0]))] # want per point
        scores = []
        
        if self.use_saved_models:
            models_list = glob.glob(os.path.join(f"{self.__saved_models_dir}/{self.__save_dir}", '[!t]*'))
            for i, shadow_model in enumerate(models_list):
                self.shadow_model = tf.keras.models.load_model(shadow_model)
                
                attack_x = attack_ds[0]
                attack_y = attack_ds[1]

                confs_scaled = self.logit_scaling(self.model_confidence(self.shadow_model, attack_x, attack_y))
                for point_index in range(len(attack_ds[0])):
                    confs_out[point_index].append(confs_scaled[point_index])
        else:
            for n in range(self.num_shadow_models):
                D_attack = self.__sampling_helper(self.data_x, self.data_y, num_samples)
                
                self.train_shadow_model(
                    D_attack[0], 
                    D_attack[1], 
                    batch_size=batch_size, 
                    epochs=training_epochs, 
                    lr=lr, 
                    shadow_model_number=n+1,
                    adam=False if self.sgd else True
                )
                attack_x = attack_ds[0]
                attack_y = attack_ds[1]

                confs_scaled = self.logit_scaling(self.model_confidence(self.shadow_model, attack_x, attack_y))
                for point_index in range(len(attack_ds[0])):
                    confs_out[point_index].append(confs_scaled[point_index])
                    
                self.shadow_model = tf.keras.models.clone_model(
                    self.shadow_model, input_tensors=None, clone_function=None
                )

        observed_confidence = self.logit_scaling(self.model_confidence(self.model, attack_x, attack_y))
        for i, shadow_confs in enumerate(tqdm(confs_out, desc='Attacking w/ LiRA: Patience my child xD')):
            shadow_confs = torch.Tensor(shadow_confs)
            shadow_confs = shadow_confs[torch.isfinite(shadow_confs)]
            mean_out = torch.mean(shadow_confs)
            var_out = torch.var(shadow_confs)
            
            score = norm.cdf(observed_confidence[i], loc=mean_out, scale=var_out+1e-30)
            scores.append((1 - score, shadow_confs))
    
        return scores
        
        
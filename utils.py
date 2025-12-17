import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    
    true_labels = np.array(real_labels)
    predicted_labels = np.array(predicted_labels) 
    
    true_positive = np.sum( ( true_labels == 1 ) & ( predicted_labels == 1) ) 
    true_negative = np.sum( ( true_labels == 0) & ( predicted_labels == 0) ) 
    
    false_positive = np.sum( (true_labels == 0) & ( predicted_labels == 1) ) 
    false_negative = np.sum( (true_labels == 1) & ( predicted_labels == 0) )
    
    if (true_positive + false_positive == 0):
        return 0.0 
        
    precision =  true_positive / ( true_positive + false_positive) 
    recall = true_positive / ( true_positive + false_negative) 
    
    if (precision + recall == 0):
        return 0.0 
        
    f1_score = 2 * precision * recall / ( precision + recall)
    
    return float(f1_score)
    


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        point1 = np.array(point1)
        point2 = np.array(point2) 
        
        return np.sum( np.abs(point1 - point2) ** 3 ) ** (1/3)
    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        point1 = np.array(point1)
        point2 = np.array(point2) 
        
        return np.sum( (point1 - point2) ** 2 ) ** 0.5 

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        point1 = np.array(point1)
        point2 = np.array(point2) 
        
        norm1 = np.linalg.norm(point1)
        norm2 = np.linalg.norm(point2)
        
        dot_product = np.dot( point1 , point2) 
        
        if ( norm1 == 0) or ( norm2 == 0):
            return 1.0 
        else:
            cosine = dot_product / (norm1 * norm2)
            
            return 1 - cosine 


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[float]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[float]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        distance_functions = list(distance_funcs.keys())
        best_f1 = -np.inf
        
        for k in range(1 , 30 , 2):
            for distance_function_name , distance_function in distance_funcs.items():                
                knn = KNN( k , distance_function )
                knn.train(x_train , y_train) 
                
                y_val_pred = knn.predict(x_val) 
                
                f1 = f1_score(y_val , y_val_pred) 
                
                if f1 > best_f1:
                    best_f1 = f1 
                    self.best_k = k 
                    self.best_distance_function = distance_function_name 
                    self.best_model = knn 
                elif f1 == best_f1:
                    
                    if self.best_distance_function is None or  ( distance_functions.index(distance_function_name) <   distance_functions.index(self.best_distance_function) ) : 
                        
                        best_f1 = f1 
                        self.best_k = k 
                        self.best_distance_function = distance_function_name
                        self.best_model = knn 
                    elif ( distance_functions.index(distance_function_name) == distance_functions.index(self.best_distance_function) ) and (k < self.best_k) :
                        best_f1 = f1 
                        self.best_k = k 
                        self.best_distance_function = distance_function_name
                        self.best_model = knn 

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them. 
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[float]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[float]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
        
        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """
        
        # You need to assign the final values to these variables
        
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        
        distance_func_names = list(distance_funcs.keys())
        scalar_func_names = list(scaling_classes.keys()) 
        
        best_f1 = -np.inf
        
        for k in range(1 , 30 , 2):
            for scaling_name , scalar_function in scaling_classes.items():
                
                scaler = scalar_function()
                x_train_scaled = scaler(x_train)
                x_val_scaled = scaler( x_val) 
                
                for distance_function_name , distance_function in distance_funcs.items():
                    
                    knn = KNN( k , distance_function )
                    knn.train(x_train_scaled , y_train )
                    
                    y_pred_val = knn.predict( x_val_scaled ) 
                    f1 = f1_score( y_val , y_pred_val)
                    
                    if f1 > best_f1:
                        best_f1 = f1 
                        self.best_k = k 
                        self.best_scaler = scaling_name
                        self.best_distance_function = distance_function_name
                        self.best_model = knn 
                    elif f1 == best_f1:
                        if ( scalar_func_names.index(scaling_name) < scalar_func_names.index(self.best_scaler) ):
                            best_f1 = f1 
                            self.best_k = k 
                            self.best_distance_function = distance_function_name
                            self.best_scaler = scaling_name 
                            self.best_model = knn 

                        elif ( scalar_func_names.index(scaling_name) == scalar_func_names.index(self.best_scaler) ) :
                            if ( distance_func_names.index(distance_function_name) < distance_func_names.index(self.best_distance_function)): 
                                best_f1 = f1 
                                self.best_k = k 
                                self.best_distance_function = distance_function_name
                                self.best_scaler = scaling_name 
                                self.best_model = knn 

                            elif (distance_func_names.index(distance_function_name) == distance_func_names.index(self.best_distance_function)) and (k< self.best_k) :
                                best_f1 = f1 
                                self.best_k = k 
                                self.best_distance_function = distance_function_name
                                self.best_scaler = scaling_name 
                                self.best_model = knn
        


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        normalised_features = []
        
        for feature in features:
            
            feature = np.array(feature)
            norm = np.linalg.norm(feature)
            if norm == 0:
                normalised_features.append(feature.tolist())
            else:
                normalised_features.append((feature / norm).tolist())
                    
        return normalised_features


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        features = np.array(features, dtype=float)
        
        mins = features.min(axis=0)
        maxs = features.max(axis=0)
        ranges = maxs - mins
        
        ranges[ranges == 0] = 1
        
        normalized = (features - mins) / ranges
        
        return normalized.tolist()
    

"""
        LOF TEST FOR OUTLIER DETECTION

    Summary:
Ordering Points To Identify the Clustering Structure (OPTICS) is an algorithm for finding density-based clusters in spatial data. Its basic idea is similar to DBSCAN, but it addresses one of DBSCAN's major weaknesses: the problem of detecting meaningful clusters in data of varying density.

"""

from typing import List, Tuple
from numpy.core.defchararray import array
from numpy.core.numeric import count_nonzero
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils import _weight_vector
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from os import system

import sys, os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

# print(__doc__)
_ = system('cls')

### Constants
VARIABLES_USED = ['length', 'width', 'height', 'weight']

COLORS_PALETTE = {'axis2' : ['g.', 'r.', 'b.', 'y.', 'c.']
                  , 'axis3' : ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
                  , 'axis4' : ['g.', 'm.', 'y.', 'c.']}

DISPLAY_MODE = {'axis2' : '3d'
                , 'axis3' : '3d'
                , 'axis4' : '3d'}

# Sortable ASINs must be all below these values
SORT_LIMITS = {'length': 17.91
               , 'width': 13.39
               , 'height': 10.41
               , 'weight': 27.16}

DEFAULT_CASES = np.array([[7.0, 8.5, 0.8, 2.0],
                        [5.0, 11.0, 3.0, 3.0],
                        [12.0, 12.0, 12.0, 5.0]])  

# HYPERPARAMETERS
MIN_SAMPLES = 200                 # INITIAL: 50
MIN_CLUSTER_SIZE = 0.1            # INITIAL: 0.05
SMALL_EPS = 1.2
HIGH_EPS = 1.8
NUMBER_OF_ITEMS = 5000
N_NEIGHBORS = 500
CONTAMINATION_RATIO = 0.01        # To avoid overfitting, do not put below 0.01

EX_TYPE = 'book_data'             # Choose X to be book_data, 2d or 3d in the source
N_POINTS_PER_CLUSTER = 250

CM_TO_IN = 0.393701
KG_TO_LBS = 2.20462

### Functions
def rect_prism(x_range: array, y_range: array, z_range: array, ax: Axes3D):
    """Plot the sortability prism on the figure

    Args:
        x_range (array): From 0 to width's limit
        y_range (array): From 0 to width's limit
        z_range (array): From 0 to width's limit
        ax (Axes3D): Current activated axis
    """

    # x_y_edge
    xx, yy = np.meshgrid(x_range, y_range)
    for value in [0, 1]:
        ax.plot_surface(xx, yy, z_range[value] * np.ones((xx.shape)), color="g", alpha=0.2)

    # y_z_edge
    yy, zz = np.meshgrid(y_range, z_range)
    for value in [0, 1]:
        ax.plot_surface(x_range[value] * np.ones((yy.shape)), yy, zz, color="g", alpha=0.2)

    # x_z_edge
    xx, zz = np.meshgrid(x_range, z_range)
    for value in [0, 1]:
        ax.plot_surface(xx, y_range[value] * np.ones((xx.shape)), zz, color="g", alpha=0.2)


def test_outlier(X_new: np.array, model_outlier_detector: LocalOutlierFactor, X: np.array) -> Tuple[np.array, np.array]:
    """Checking whether the trained LOF model considers the new ASIN as an outlier (does it come from the same distribution?) considering dimensions AND weight
    and an unsortable item, above the limits

    Args:
        X_new (np.array): New array of new data points. Used to compute the negative outlier factor wrt the threshold of the model. Could be one or more points.
        model_outlier_detector (LocalOutlierFactor): Trained model.
        X (np.array): Training set not cleaned. Noise in the training is considered with the CONTAMINATION_RATIO

    Returns:
        pred_new (np.array(int)): Prediction on whether new points are considered outliers or not (-1 yes they are, 1 they are not and hence come from same distribution)
        sortable_new (np.array(Boolean)): Discrimination of X_new between sortable (True) or unsortable (False)
    """
    lof = LocalOutlierFactor(n_neighbors=N_NEIGHBORS, contamination=CONTAMINATION_RATIO, novelty=True)
    lof.fit(X)

    X_new = X_new.reshape(-1, len(VARIABLES_USED))

    # Discriminate based on likely to be an outlier
    pred_new = lof.predict(X_new)
    scores = lof.decision_function(X_new)       # This is the shifted opposite of the LOF of X i.e., the bigger values correspond to inliers
    indices_outliers = np.asarray(np.where(pred_new == -1))
    n_outlier = pred_new[pred_new == -1].size

    # Discriminate based on sortability criteria
    sortable_new = []
    for X_item in X_new[:, range(len(VARIABLES_USED)-1)]:
        if (all(X_item < SORT_LIMITS['length']) 
            and (sum(X_item < SORT_LIMITS['width']) >= 2)
            and (sum(X_item < SORT_LIMITS['height']) >= 1)
            and (X_item[-1] < SORT_LIMITS['weight'])):
            sortable_new.append(True)
        else:
            sortable_new.append(False)

    indices_nonsort = [i for i, x in enumerate(sortable_new) if not x] 
    n_nonsort = len(sortable_new) - count_nonzero(sortable_new)

    indices_revision = np.intersect1d(indices_outliers, indices_nonsort)

    print('\nAmong the {} points tested, {} are outliers.'.format(len(X_new), n_outlier))
    print('These outliers are those in positions: %s' % (indices_outliers,))
    print('Whose shifted opposite LOF scores are %s' % scores + ' respectively (the more negative, the more an outlier it is)')

    print('\nOn the other hand, {} are non-sortable.'.format(n_nonsort))
    print('These outliers are those in positions: %s' % (indices_nonsort,))
    
    print('\nIn conclusion, outliers in position {} need further revision'.format(indices_revision))
    
    return pred_new, np.array(sortable_new)


def plot_result(X_new: np.array, pred_new: tuple, sortable_new: tuple, X: np.array):
    """Plotting the CLEANED training set along with the new data point, varying in color depending if it is an outlier, a non-sort object or both

    Args:
        X_new (np.array):  New data point. Used to compute the negative outlier factor wrt the threshold of the model.
        pred_new (np.array(int)): Prediction on outlier (-1 yes they are, 1 they are not and hence come from same distribution)
        sortable_new (np.array(Boolean)): Discrimination of X_new between sortable (True) or unsortable (False)
        X (np.array): Training set. Useful for plotting along with the outlier. It corresponds to cleanded data set.
    """
    plt.figure()
    ax = plt.axes(projection ="3d")

    # Plot cleaned training set
    ax.scatter(X[:, 0], X[:, 1], zs = X[:, 2], c = 'b', marker = '.', alpha=0.05)

    # Plot new data   
    # X_new can be of many OR one data point --> not possible to vectorize, since 1D array cannot be indexed as 2D. Need to differentiate cases
    if len(X_new.flatten()) == len(VARIABLES_USED):
        # Inlier
        if pred_new != -1 and sortable_new:
            ax.scatter(X_new[0], X_new[1], X_new[2], c='g', marker='+', alpha=0.8, s=100)
        # Sortable and outlier
        elif pred_new == -1 and sortable_new:
            ax.scatter(X_new[0], X_new[1], X_new[2], c='orange', marker='+', alpha=0.8, s=100)
        # Non-sortable
        else:
            ax.scatter(X_new[0], X_new[1], X_new[2], c='r', marker='+', alpha=0.8, s=100)
    else:
        # Inlier
        ax.scatter(X_new[(pred_new != -1) & sortable_new, 0], X_new[(pred_new != -1) & sortable_new, 1], X_new[(pred_new != -1) & sortable_new, 2], c='g', marker='+', alpha=0.8, s=100)

        # Sortable and outlier
        ax.scatter(X_new[(pred_new == -1) & sortable_new, 0], X_new[(pred_new == -1) & sortable_new, 1], X_new[(pred_new == -1) & sortable_new, 2], c='orange', marker='+', alpha=0.8, s=100)

        # Non-sortable
        ax.scatter(X_new[np.invert(sortable_new), 0], X_new[np.invert(sortable_new), 1], X_new[np.invert(sortable_new), 2], c='r', marker='+', alpha=0.8, s=100)

    ### Plot boundary surfaces
    rect_prism(np.array([0, SORT_LIMITS['width']]),
               np.array([0, SORT_LIMITS['length']]),
               np.array([0, SORT_LIMITS['height']]),
               ax)


    ax.tick_params(labelsize = 7)
    ax.set_xlabel('Width')
    ax.set_ylabel('Length')
    ax.set_zlabel('Height')

    ax.set_title('Novelty detection using \nLOF algorithm (c_ratio=' + str(CONTAMINATION_RATIO) + ')', fontsize = 20)

    plt.tight_layout()
    plt.show()


def subplot(number: int, axis_name: plt.subplot, X: np.array, labels: np.array, mode: str = '2d'):
    """This function intakes all the input in order to be able to draw a subplot in the final clustering digram

    Args:
        number (int): This number is an identifier on the axis name
        axis_name (plt.subplot): Variable which holds the information on a given subplot    
        X (np.array): Data coordinates in a tuple (x, y, z) 
        labels (np.array): Classification in clusters by the algorithm
        mode (string): euther in '2d' or in '3d'. Defaults to '2d'.
        
    Raises:
        Exception: Happens when mode has not been selected
    """
    unique_lab = len(set(labels))

    if mode == '2d':
        for klass, color in zip(range(0, unique_lab), COLORS_PALETTE['axis' + str(number)]):
            Xk = X[labels == klass]
            axis_name.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
        axis_name.plot(X[labels == -1, 0], X[labels == -1, 1], 'k+', alpha=0.1)
        axis_name.tick_params(labelsize = 7)
        axis_name.set_xlabel('Width')
        axis_name.set_ylabel('Length')
        
    elif mode == '3d':
        for klass, color in zip(range(0, unique_lab), COLORS_PALETTE['axis' + str(number)]):
            Xk = X[labels == klass]
            axis_name.scatter(Xk[:, 0], Xk[:, 1], zs = Xk[:, 2], c = color.replace('.',''), marker = '.')
        axis_name.scatter(X[labels == -1, 0], X[labels == -1, 1], X[labels == -1, 2], c='k', marker='.', alpha=0.1)
        axis_name.tick_params(labelsize = 7)
        axis_name.set_xlabel('Width')
        axis_name.set_ylabel('Length')
        axis_name.set_zlabel('Height')

    else:
        raise Exception("Try a proper representation mode")


### MAIN
# Generate data
def main():
    np.random.seed(0)
    if EX_TYPE == '2d':
        n_dims = 2
        C1 = [-5, -2] + .8 * np.random.randn(N_POINTS_PER_CLUSTER, n_dims)
        C2 = [4, -1] + .1 * np.random.randn(N_POINTS_PER_CLUSTER, n_dims)
        C3 = [1, -2] + .2 * np.random.randn(N_POINTS_PER_CLUSTER, n_dims)
        C4 = [-2, 3] + .3 * np.random.randn(N_POINTS_PER_CLUSTER, n_dims)
        C5 = [3, -2] + 1.6 * np.random.randn(N_POINTS_PER_CLUSTER, n_dims)
        C6 = [5, 6] + 2 * np.random.randn(N_POINTS_PER_CLUSTER, n_dims)
        X = np.vstack((C1, C2, C3, C4, C5, C6))

    elif EX_TYPE == '3d':
        n_dims = 3
        C1 = [-5, -2, 1] + .8 * np.random.randn(N_POINTS_PER_CLUSTER, n_dims)
        C2 = [4, -1, 3] + .1 * np.random.randn(N_POINTS_PER_CLUSTER, n_dims)
        C3 = [1, -2, 2] + .2 * np.random.randn(N_POINTS_PER_CLUSTER, n_dims)
        C4 = [-2, 3, 1] + .3 * np.random.randn(N_POINTS_PER_CLUSTER, n_dims)
        C5 = [3, -2, 0] + 1.6 * np.random.randn(N_POINTS_PER_CLUSTER, n_dims)
        C6 = [5, 6, 3] + 2 * np.random.randn(N_POINTS_PER_CLUSTER, n_dims)
        X = np.vstack((C1, C2, C3, C4, C5, C6))

    elif EX_TYPE == 'book_data':
        filedir = os.path.dirname(__file__)
        datadir = os.path.join(filedir, 'data')
        raw_data = pd.read_csv(datadir + r'\book_data.csv', sep = "\t", nrows = NUMBER_OF_ITEMS)
        # url_drive = 'https://drive.google.com/file/d/1ZD6I_cZ7DZXWveSYnxBu2_m-7sS7BTzt/view?usp=sharing'
        # raw_data = pd.read_csv(url_drive, sep = "\t", nrows = NUMBER_OF_ITEMS)
        
        # Cleaning and slicing
        if raw_data.isna().sum().sum() < .10 * raw_data.size: 
            raw_data = raw_data.dropna()
        else:
            raise Exception("Careful! Deleting NaN values would cut most of the dataset")

        X = pd.DataFrame(raw_data, columns= ['isbn', 'asin', 'item_name', 'gl_name', 'product_type', 'pkg_height', 'pkg_width', 'pkg_length', 'pkg_dimensional_uom', 'pkg_weight', 'pkg_weight_uom'])

        # FORMATTING
        # Dimension
        if bool({'cm', 'centimeter', 'centimeters'}.intersection(list(X.pkg_dimensional_uom.unique()))):
            X.loc[((X.pkg_dimensional_uom=='cm')|(X.pkg_dimensional_uom=='centimeter')|(X.pkg_dimensional_uom=='centimeters')), ['pkg_height', 'pkg_width', 'pkg_length']] *= CM_TO_IN
            X.loc[((X.pkg_dimensional_uom=='cm')|(X.pkg_dimensional_uom=='centimeter')|(X.pkg_dimensional_uom=='centimeters')), 'pkg_dimensional_uom'] = 'inches' 

        # Weight
        if bool({'kg', 'kilo', 'kilogram', 'kilograms'}.intersection(list(X.pkg_weight_uom.unique()))):
            X.loc[((X.pkg_weight_uom=='cm')|(X.pkg_weight_uom=='centimeter')|(X.pkg_weight_uom=='centimeters')), ['pkg_height', 'pkg_width', 'pkg_length']] *= KG_TO_LBS
            X.loc[((X.pkg_weight_uom=='cm')|(X.pkg_weight_uom=='centimeter')|(X.pkg_weight_uom=='centimeters')), 'pkg_dimensional_uom'] = 'pounds' 

        # Inputs for the algorithm
        X = np.array(X.loc[:, ['pkg_width', 'pkg_length', 'pkg_height', 'pkg_weight']])  
        assert(type(X) == np.ndarray)

    # OPTICS ALGORITHM
    clust = OPTICS(min_samples = MIN_SAMPLES, min_cluster_size = MIN_CLUSTER_SIZE)
    clust.fit(X[:, range(3)])

    # DBSCAN ALGORITHM: eps = SMALL_EPS
    labels_small_eps = cluster_optics_dbscan(reachability = clust.reachability_,
                                    core_distances = clust.core_distances_,
                                    ordering = clust.ordering_, eps = SMALL_EPS)

    # DBSCAN ALGORITHM: eps = HIGH_EPS
    labels_high_eps = cluster_optics_dbscan(reachability = clust.reachability_,
                                    core_distances = clust.core_distances_,
                                    ordering = clust.ordering_, eps = HIGH_EPS)

    space = np.arange(len(X[:, range(3)]))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]

    # LOF MODEL
    model_outlier_detector = LocalOutlierFactor(n_neighbors = N_NEIGHBORS, contamination = CONTAMINATION_RATIO) 
    y_pred = model_outlier_detector.fit_predict(X[:, range(3)])

    # filter outlier index
    outlier_index = np.where(y_pred == -1)       # -1 values are outliers
    inlier_index = np.where(y_pred != -1)        # 1 values are points from same distribution
    X_cleaned = X[inlier_index]


    ### USER CHOICE MENU
    print ("""
    ########################################################################### \n
        Hi! Welcome to the Outlier Detection tool for new Book ASINs         \n
    ########################################################################### \n
    """)

    menu = {}
    menu[1] = "Insert manually data from new book ASIN(s)."
    menu[2] = "Upload ASINs from a csv file." 
    menu[3] = "Try the algorithm with a default sample."
    menu[4] = "Plot Clustering graphs"
    menu[5] = "Exit"

    while True: 
        print("Please, select one of the options above (1-5): \n")

        options = menu.keys()
        
        for entry in options: 
            print (str(entry) + '. ' + menu[entry])

        selection = input("\nNumber selection:  ") 

        ### Case 1: Insert manually a new ASIN and test it
        if selection == '1': 
            
            control = 0
            X_new = []

            while control == 0:
                print ("\nPlease, introduce the item's width: ")
                width = float(input(''))

                print ("\nThe item's length: ")
                length = float(input(''))

                print ("\nAnd the item's height: ")
                height = float(input(''))

                print ("""
                \nPlease, introduce now the unit of measurement from last measurements:
                1. Inches
                2. Centimeters
                """)
                uom_dim = int(input('Enter one of the numbers (1 or 2): '))

                if uom_dim == 2:
                    (width, length, height) = (width, length, height) * CM_TO_IN

                print ("\nPlease, introduce the item's weight: ")
                weight = float(input(''))

                print ("""
                \nPlease, introduce now the unit of measurement from last measurements:
                1. Pounds
                2. Kilograms
                """)
                uom_wgt = int(input('Enter one of the numbers (1 or 2): '))

                if uom_wgt == 2:
                    weight *= KG_TO_LBS

                X_new.append(np.array([width, length, height, weight]))
                # X_new = np.array([width, length, height, weight])

                print('\nWould you like to add a new ASIN?: ')
                more_asins_user = input('y/n: ')
                
                if more_asins_user == 'n':
                    control = 1
            
            # Formatting
            X_new = np.asarray(X_new)

            # Test the new data
            print('\nASIN(s) info registered. Now testing...\n')
            pred_new, sortable_new = test_outlier(X_new, model_outlier_detector, X)

            # Plotting results
            print ("\nWould you like to see the plotting of the result? ")
            plot_flag = input('y/n: ')
            if plot_flag == 'y':
                plot_result(X_new, pred_new, sortable_new, X_cleaned)
            

        ### Case 2: Try algorithm with loaded cases from a csv file
        elif selection == '2':
            
            print("""
            #################################################################
                CAUTION: 
            - Csv file should be on the same folder as this script
            - Data should be separated by a semi-colon
            - Please, remove headers but provide data in the following order: 
                        (width, length, height, weight)
            #################################################################
            """)
            print('\nPlease, provide the name of your file : ')
            filename = input('')

            if filename[-3:] != 'csv':
                filename += '.csv'

            test = os.path.join(filedir, 'data', filename)
            X_new = pd.read_csv(os.path.join(filedir, 'data', filename), sep = ";", header = None)
            
            # Cleaning and slicing
            if X_new.isna().sum().sum() < .10 * X_new.size: 
                X_new = X_new.dropna()
            else:
                raise Exception("Careful! Deleting NaN values would cut most of the data")

            X_new.columns = ['pkg_width', 'pkg_length', 'pkg_height', 'pkg_weight']

            # FORMATTING
            # Dimension
            print ("""
            \nPlease, introduce now the unit of measurement from last measurements:
            1. Inches
            2. Centimeters
            """)
            uom_dim = int(input('Enter one of the numbers (1 or 2): '))

            if uom_dim == 2:
                X_new.loc[:, ['pkg_height', 'pkg_width', 'pkg_length']] *= CM_TO_IN

            print ("""
            \nPlease, introduce now the unit of measurement from last measurements:
            1. Pounds
            2. Kilograms
            """)
            uom_wgt = int(input('Enter one of the numbers (1 or 2): '))

            if uom_wgt == 2:
                X_new.loc[:, ['pkg_weight']] *= KG_TO_LBS
            
            # Inputs for the clustering algorithm
            X_new = np.array(X_new.loc[:,['pkg_width', 'pkg_length', 'pkg_height', 'pkg_weight']])  
            assert(type(X_new) == np.ndarray)
            
            # Test the sample data
            print('\nASIN(s) info registered. Now testing...\n')
            pred_new, sortable_new = test_outlier(X_new, model_outlier_detector, X)

            # Plotting results
            print ("\nWould you like to see the plotting of the result? \n")
            plot_flag = input('y/n: ')
            if plot_flag == 'y':
                plot_result(X_new, pred_new, sortable_new, X_cleaned)
                        

        ### Case 3: Try algorithm with pre-filled cases
        elif selection == '3': 
            print ("""
            \nPlease, introduce one of the default cases to see the outcome plotted:
            1. An Inlier
            2. A close Outlier
            3. A distant Outlier
            4. A mix of them
            5. Exit
            """)
            default_case = int(input('Enter one of the numbers (1 - 5): '))

            if default_case == 1:
                X_new = DEFAULT_CASES[0]

            elif default_case == 2:
                X_new = DEFAULT_CASES[1]

            elif default_case == 3:
                X_new = DEFAULT_CASES[2]

            elif default_case == 4:
                X_new = DEFAULT_CASES 

            elif default_case == 5:
                break

            # Test the sample data
            print('\nASIN(s) samples registered. Now testing...\n')
            pred_new, sortable_new = test_outlier(X_new, model_outlier_detector, X)

            # Plotting results
            print ("\nWould you like to see the plotting of the result? ")
            plot_flag = input('y/n: ')
            if plot_flag == 'y':
                plot_result(X_new, pred_new, sortable_new, X_cleaned)
            

        ### Case 4: Plot clustering graphs
        elif selection == '4':

            ### PLOTTING FIRST FIGURE
            plt.figure(0)
            G = gridspec.GridSpec(2, 3)
            ax1 = plt.subplot(G[0, :])
            if DISPLAY_MODE['axis2'] == '2d':
                ax2 = plt.subplot(G[1, 0])
            elif DISPLAY_MODE['axis2'] == '3d':
                ax2 = plt.subplot(G[1, 0], projection=DISPLAY_MODE['axis2'])

            if DISPLAY_MODE['axis3'] == '2d':   
                ax3 = plt.subplot(G[1, 1])
            elif DISPLAY_MODE['axis3'] == '3d':
                ax3 = plt.subplot(G[1, 1], projection=DISPLAY_MODE['axis3'])

            if DISPLAY_MODE['axis4'] == '2d':
                ax4 = plt.subplot(G[1, 2])
            elif DISPLAY_MODE['axis4'] == '3d':
                ax4 = plt.subplot(G[1, 2], projection=DISPLAY_MODE['axis4'])

            # Axis 1: Reachability plot
            colors = ['g.', 'r.', 'b.', 'y.', 'c.']
            unique_opt = len(set(labels))
            for klass, color in zip(range(0, unique_opt), colors):
                Xk = space[labels == klass]
                Rk = reachability[labels == klass]
                ax1.plot(Xk, Rk, color, alpha=0.3)
            ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
            ax1.plot(space, np.full_like(space, HIGH_EPS, dtype=float), 'k-.', alpha=0.3)
            ax1.plot(space, np.full_like(space, SMALL_EPS, dtype=float), 'k-.', alpha=0.3)
            ax1.set_ylabel('Reachability (epsilon distance)', fontsize = 6)
            ax1.set_title('Reachability Plot', fontsize = 12)
            ax1.tick_params(labelsize = 7)

            # Axis 2: OPTICS
            subplot(2, ax2, X, clust.labels_, mode = DISPLAY_MODE['axis2'])
            ax2.set_title('Automatic Clustering\nOPTICS  (' + str(MIN_SAMPLES) + ', ' + str(MIN_CLUSTER_SIZE) + ')', fontsize = 10)

            # Axis 3: DBSCAN at SMALL EPS
            subplot(3, ax3, X, labels_small_eps, mode = DISPLAY_MODE['axis3'])
            ax3.set_title('Clustering at ' + str(SMALL_EPS) + ' epsilon cut\nDBSCAN', fontsize = 10)

            # Axis 4: LOF Algorithm, the one performing the novelty detection
            ax4.scatter(X[inlier_index, 0], X[inlier_index, 1], zs = X[inlier_index, 2], c = "g", s = 65, marker = '.', alpha = 0.3)
            ax4.scatter(X[outlier_index, 0], X[outlier_index, 1], X[outlier_index, 2], c='k', marker='+', alpha=0.8)
            ax4.tick_params(labelsize = 7)
            ax4.set_xlabel('Width')
            ax4.set_ylabel('Length')
            ax4.set_zlabel('Height')
            ax4.set_title('Anomaly detection using \nLOF algorithm (c_ratio=' + str(CONTAMINATION_RATIO) + ')', fontsize = 10)

            print('Algorithm used for outlier detection: Local Outlier Factor \n')
            print('Number of counted outliers: ')
            print(np.count_nonzero(model_outlier_detector.negative_outlier_factor_ < model_outlier_detector.offset_))

            plt.tight_layout()
            plt.show()


        ### Case 5: Exit the tool
        elif selection == '5': 
            print('\nThanks for using the tool! Please share\n\n')
            break

        else: 
            print ("Unknown Option Selected!\n") 

if __name__ == '__main__':
    main()
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# from matplotlib import style

# import pandas as pd
# from pandas.plotting import scatter_matrix

from pdpbox import pdp, info_plots

from alepython import ale


def plot_corr_matrix(corr_matrix):
    
    fig = plt.figure(figsize = (10,10))
    ax1 = plt.subplot(111)
    im = ax1.matshow(np.array(corr_matrix), cmap = plt.cm.binary)

    ax1.set_xticks(np.arange(len(corr_matrix.columns)))
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_yticks(np.arange(len(corr_matrix.columns)))
    ax1.set_ylim(len(corr_matrix.columns) - 0.5, -0.5)

    ax1.set_xticklabels(corr_matrix.columns, rotation = 75)
    ax1.set_yticklabels(corr_matrix.columns)

    ax1.grid(False, which = 'both')
    plt.title('Correlation matrix')

    plt.colorbar(im, fraction = 0.046, pad = 0.04)

#Break up into smaller funtions, remove plt.show()     
# def plot_precision_recall_curves(classifiers):
    # n_classes = len(classifiers.keys())
    # i = 0
    # ax1 = {}
    # ax2 = {}
    # fig = plt.figure(figsize = (15,n_classes*6))
    # for clf in classifiers.keys():
        # index_over_95 = len(recall[clf][recall[clf] >=0.95]) -1

        # if i == 0:
            # ax1[i] = plt.subplot2grid((n_classes,2),(i,0), rowspan = 1, colspan = 1)
        # else:
            # ax1[i] = plt.subplot2grid((n_classes,2),(i,0), rowspan = 1, colspan = 1, sharex = ax1[0])
        # ax1[i].plot(threshold[clf],precision[clf][:-1], 
            # color = 'g', label = 'Precision')
        # ax1[i].plot(threshold[clf],recall[clf][:-1], 
            # color = 'b', label = 'Recall')
        # ax1[i].plot([threshold[clf][index_over_95],threshold[clf][index_over_95]], [0, recall[clf][index_over_95]]
                    # ,'k:')
        # ax1[i].plot([ 0,threshold[clf][index_over_95] ] 
                     # ,[ recall[clf][index_over_95], recall[clf][index_over_95] ]
                     # ,'k:')

        # ax1[i].annotate( '%0.3f' %precision[clf][index_over_95],(threshold[clf][index_over_95],precision[clf][index_over_95])
                       # ,xytext = (threshold[clf][index_over_95] + 0.15,precision[clf][index_over_95])
                       # ,arrowprops = dict(facecolor = 'yellow', edgecolor = 'grey'))


        # plt.xlabel('Threshold')
        # plt.legend()
        # min_y_coord = min(precision[clf])
        # plt.axis([0,1,min_y_coord,1.01])
        # plt.title(clf)

        # if i == 0:
            # ax2[i] = plt.subplot2grid((n_classes,2), (i,1), rowspan = 1, colspan = 1, sharey = ax1[i])
        # else:
            # ax2[i] = plt.subplot2grid((n_classes,2), (i,1), rowspan = 1, colspan = 1, sharex = ax2[0], sharey = ax1[i])
        # ax2[i].plot(recall[clf][:-1], precision[clf][:-1], color = 'r')

        # plt.axis([0,1,min_y_coord,1.01])
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title(clf)


        # i += 1

        # plt.grid(True)


    # plt.show()
     
     
     
# def plot_1d_pdp(model, X,y = None, model_features = None, feature = None, **kwargs):
    
    # if y is not None:
        # model.fit(X,y)
        
    # pdp_plt = pdp.pdp_isolate(model = model,dataset =  X,model_features = model_features, 
                              # feature = feature)
    
    # pdp.pdp_plot(pdp_plt, feature, **kwargs)

def plot_ale_plot(model, X, X_unscaled = None, features = None, **kwargs):
    
    '''
    Plots an ale plot. If features contains 1 feature, it produces a 1 dimensional ale plot, 
    if it contains 2 features it produces a 2 dimensional ALE plot
    '''
    fig, ax = plt.subplots()
    ale.ale_plot(model, X, features,ax = ax, **kwargs)

    
    
    if X_unscaled is not None:
        meanx = X_unscaled[features[0]].mean()
        stdx = X_unscaled[features[0]].std()
        #Unscale x values
        def unscale_xticks(x, pos):
            return ('%.0f' %(x*stdx + meanx))

        if len(features) == 2:

            meany = X_unscaled[features[1]].mean()
            stdy = X_unscaled[features[1]].std()
            #Unscale y values
            def unscale_yticks(x, pos):
                return ('%.0f' %(x*stdy + meany))     
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(unscale_yticks))
            
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(unscale_xticks))
                
    return fig, ax    
    
    
def plot_1d_pdp(model,X,y = None, X_unscaled = None,model_features = None, feature = None, **kwargs):

    '''
    Plots a 1d pdp plot with the x-axis being unscaled.
    
    X_scaled: A pandas dataframe or numpy array.
    Contains the unscaled values of X.
    
    All other variables are the same as for plot_1d_pdp()
    '''    
    if y is not None:
        model.fit(X,y)
    
    pdp_plt = pdp.pdp_isolate(model = model,dataset =  X,model_features = model_features, 
                              feature = feature)
    
    fig, ax = pdp.pdp_plot(pdp_plt, feature, **kwargs)

    
    if X_unscaled is not None:
        mean = X_unscaled[feature].mean()
        std = X_unscaled[feature].std()
        #Unscale x values
        def unscale_ticks(x, pos):
            return ('%.1f' %(x*std + mean))

        ax['pdp_ax']['_pdp_ax'].xaxis.set_major_formatter(mticker.FuncFormatter(unscale_ticks))

    return fig, ax
    
# def plot_2d_pdp(model, X,y = None, model_features = None, features = None, **kwargs):
    
    # if y is not None:
        # model.fit(X,y)
        
    # pdp_plt = pdp.pdp_interact(model = model,dataset =  X,model_features = model_features, 
                              # features = features)
    
    # pdp.pdp_interact_plot(pdp_plt, feature_names = features, **kwargs)
    
    
def plot_2d_pdp(model, X,y = None,X_unscaled = None, model_features = None, features = None, **kwargs):

    '''
    Plots a 1d pdp plot with the x-axis being unscaled.
    
    X_scaled: A pandas dataframe or numpy array.
    Contains the unscaled values of X.
    
    All other variables are the same as for plot_1d_pdp()
    '''        
    
    if y is not None:
        model.fit(X,y)
        
        
    pdp_plt = pdp.pdp_interact(model = model,dataset =  X,model_features = model_features, 
                            features = features)
    
    fig, ax = pdp.pdp_interact_plot(pdp_plt, feature_names = features, **kwargs)
    
    if X_unscaled is not None:
        meanx = X_unscaled[features[0]].mean()
        stdx = X_unscaled[features[0]].std()
        #Unscale x values
        def unscale_xticks(x, pos):
            return ('%.1f' %(x*stdx + meanx))
        
        meany = X_unscaled[features[1]].mean()
        stdy = X_unscaled[features[1]].std()
        #Unscale y values
        def unscale_yticks(x, pos):
            return ('%.1f' %(x*stdy + meany))

        ax['pdp_inter_ax'].xaxis.set_major_formatter(mticker.FuncFormatter(unscale_xticks))
        ax['pdp_inter_ax'].yaxis.set_major_formatter(mticker.FuncFormatter(unscale_yticks))
        
    return fig, ax
 
def plot_predicted_curve(regr, axes, ax = None, show = False, **kwargs):
    
    if ax is None:
        fig = plt.figure(figsize = (8,5))
        ax = fig.gca()
    else:
        fig = ax.get_figure()
            
    x0s = np.linspace(axes[0], axes[1], 100)
    y_pred = regr.predict(x0s.reshape(-1,1))
    ax.plot(x0s, y_pred, **kwargs)    
    
    if show:
        plt.show()
        
def plot_pr_curves(model, precision, recall, threshold, ax = None, show = False):
    
    if ax is None:
        fig = plt.figure(figsize = (8,5))
        ax = fig.gca()
    else:
        fig = ax.get_figure()
    
    ax.plot(threshold,precision[:-1], 
        color = 'g', label = 'Precision')
    ax.plot(threshold,recall[:-1], 
        color = 'b', label = 'Recall')
    
    min_y_coord = max(min(precision), min(recall))

    if type(model).__name__ == 'BaggingClassifier':
        plt.title('Precision-Recall curve for %s' %(type(model.base_estimator_).__name__))       
    else:
        plt.title('Precision-Recall curve for %s' %(type(model).__name__))
   
    plt.axis([0,1,min_y_coord, 1.01])
    plt.legend()
    plt.xlabel('Threshold')
    
    if show:
        plt.show()
        
        
def plot_precision_recall_curve(model,precision, recall, ax = None, show = False):
   
    if ax is None:
        fig = plt.figure(figsize = (8,5))
        ax = fig.gca()
    else:
        fig = ax.get_figure()
  
    ax.plot(recall[:-1], precision[:-1], color = 'r')   
    
    min_y_coord = min(precision)
    if type(model).__name__ == 'BaggingClassifier':
        plt.title('Precision-Recall curve for %s' %(type(model.base_estimator_).__name__))       
    else:
        plt.title('Precision-Recall curve for %s' %(type(model).__name__))
        
    plt.axis([0,1,min_y_coord,1.01])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    if show:
        plt.show()
 

     
if __name__ == '__main__':
    pass


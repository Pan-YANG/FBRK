from __future__ import division
import sys
sys.path.append("***") # enter the system path

from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize
import scipy
import os
import matplotlib.pyplot as plt

def nearSPD_R(A,folder):
    """
    function to calculate the closest symmetric positive definitive matrix of A, by calling a R function
    :param A: the input matrix, should be an array
    :param folder: the folder where the working file is located
    :return: the output SPD matrix, should be an array
    """
    A_flat=A.flatten().T
    A_flat = pd.DataFrame(A_flat, columns=['x'])
    A_flat.to_csv(folder+'nearSPD.csv', index=False)
    os.system('"Rscript.exe" --nosave nearSPD.R')
    Ahat = pd.read_csv(folder + 'nearSPD.csv', sep=',', usecols=[1])
    Ahat = Ahat.as_matrix()
    Ahat = np.asarray(Ahat)
    Ahat = Ahat.reshape((A.shape[0],A.shape[1]))
    return Ahat

def rain_field_1d_to_2d(rain_field,rain_field_var,col_no,):
    """
    function to convert 1d array of rianfall retrived from .csv file to 2d rainfall fields
    :param rain_field: the retrived 1d rainfall, pandas DataFrame
    :param rain_field_var: the retrived 1d rainfall variance, pandas DataFrame
    :param col_no: the column number in rain_field and rain_field_var
    :return: returns rain_field_out, rain_field_out_0 and rain_field_var_out, both are numpy matrix
    """
    rain_field_out=np.asmatrix(rain_field)[:,col_no]
    rain_field_out=rain_field_out.reshape((100,200)).transpose()
    rain_field_var_out=np.asmatrix(rain_field_var)[:,col_no]
    rain_field_var_out=rain_field_var_out.reshape((100,200)).transpose()
    rain_field_out_0=rain_field_out
    rain_field_out_0[rain_field_out_0<0]=0

    return rain_field_out, rain_field_out_0, rain_field_var_out

def cal_variogram(tao,sigma,h,theta):
    "this is the function to calculate the estimated variogram"
    """
        tao,sigma,theta are the parameters in the exponential variogram model, all are tensor double scalar
        h is the distance between the points of estimate and the observed points, should be a tensor double matrix, each 
            column denotes the distances of one point of estimate. Refer to the D matrix in simple kriging
        output a theano function to calculate the exponential variogram 
        """
    variogram=tao**2+sigma**2*(1-np.exp(-h/theta))
    return variogram

def cal_dist(pos_out,pos_in):
    "the function to calculate distances between pos_out and pos_in"
    """
        pos_out and pos_in are the positions of the points to be estimate and the points already observed
        both are two column tensor matrices, with [:,0] the x position and [:,1] the y position
        output is a function to calculate the distances
        """
    m_in=pos_in.shape[0]
    m_out=pos_out.shape[0]
    pos_in_x=pos_in[:,0]
    pos_in_y=pos_in[:,1]
    pos_out_x, pos_out_y = pos_out[:,0].T, pos_out[:,1].T
    pos_in_x_1=pos_in_x.repeat(m_out,1)
    pos_in_y_1=pos_in_y.repeat(m_out,1)
    pos_out_x_2=pos_out_x.repeat(m_in,0)
    pos_out_y_2 = pos_out_y.repeat(m_in, 0)
    dist=np.sqrt(np.square(pos_in_x_1-pos_out_x_2)+np.square(pos_in_y_1-pos_out_y_2))
    return dist

def cal_krig_weight_sk(pos_out,pos_in,tao,sigma,theta,folder):
    "calculate the kriging weights, simple kriging"
    """
        pos_out and pos_in are the positions for the target output and the observed inputs
        tao,sigma, and theta are the parameters for the variogram model
        """
    pos_in=np.asmatrix(pos_in)
    pos_out=np.asmatrix(pos_out)
    n_in=pos_in.shape[0]
    n_out=pos_out.shape[0]
    C=cal_dist(pos_in,pos_in)
    C=cal_variogram(tao,sigma,C,theta)
    D=cal_dist(pos_out,pos_in)
    D=cal_variogram(tao,sigma,D,theta) # the D matrix in simple kriging
    C=(C+C.T)/2

    try:
        W = np.linalg.pinv(C) * D  # the W matrix in simple kriging
    except np.linalg.linalg.LinAlgError:
        print('+++++++++++++++++++C is singular+++++++++++++++++++++')
        # use the Tikhonov regularization if C is singular
        telta = 1000*np.finfo(np.float64).eps * np.eye(C.shape[0])
        W = scipy.linalg.pinv2(C.T * C + telta.T * telta) * C.T * D  # the W matrix in simple kriging
    except ValueError:
        print('+++++++++++++++++++C is singular+++++++++++++++++++++')
        # use the Tikhonov regularization if C is singular
        telta = np.finfo(np.float64).eps * np.eye(C.shape[0])
        W = scipy.linalg.pinv2(C.T * C + telta.T * telta) * C.T * D  # the W matrix in simple kriging

    krig_weight=W # kriging weights in simple kriging
    return krig_weight,D

def cal_krig_esti_sk(pos_out,pos_in,val_in,tao,sigma,theta,folder):
    "function to calculate the kriging estimates and the variances at target positions"
    """
        pos_out and pos_in are the positions of the output and input
        val_in are the observed values at the input positions
        tao, sigma, and theta are the parameters to calcualte variogram
        outputs are the kriging estimate and the kriging variance at the output positions
        """
    krig_weight,D=cal_krig_weight_sk(pos_out,pos_in,tao,sigma,theta,folder)
    # the kriging weights and lagrange multipliers for simple kriging
    val_in = val_in.reshape((val_in.shape[0], 1))
    try:
        krig_esti=krig_weight.T*val_in
    except ValueError:
        krig_esti=krig_weight*val_in

    var=(1-krig_weight.sum(0))*cal_variogram(tao,sigma,1000.0,theta)+np.multiply(krig_weight,D).sum(0)
    krig_sigma=np.sqrt(np.sqrt(np.square(var)))
    krig_sigma=krig_sigma.T
    return krig_esti, krig_sigma

def cal_krig_likli_fast(pos_in,val_in,covar_in,var_label,tao,sigma,theta,reg_coeff,const,IDs,folder):
    "function to calculate the likelihood of the observations given the parameters of the model"
    """
        pos_in and val_in are the positions and observed values of the inputs
        covar_in are the observed values of the covariable at pos_in
        var_label are labels indicating the level of uncertainty for the val_in at pos_in
        tao, sigma, and theta are the parameters of the variogram model
        reg_coeff, and const are the coefficients in the regression model
        """
    n_in=pos_in.shape[0]

    err_in=val_in-reg_coeff*covar_in-const
    pdfs=[]
    n_iter=int(np.ceil(n_in/3))

    for i in range(n_iter):

        if (i+1)*3>n_in:
            ID_iter=IDs[3*i:]
        else:
            ID_iter=IDs[3*i:3*i+3]

        krig_esti, krig_sigma=cal_krig_esti_sk(pos_in[ID_iter,:],np.delete(pos_in,ID_iter,0),np.delete(err_in,ID_iter).T,tao,sigma,theta,folder)
        krig_esti = np.asarray(krig_esti).ravel()
        krig_esti=reg_coeff*covar_in[ID_iter]+krig_esti+const
        krig_sigma = np.asarray(krig_sigma).ravel()
        pdfs=np.append(pdfs,scipy.stats.norm(krig_esti,np.sqrt(np.divide(np.square(krig_sigma),scipy.special.expit(val_in[ID_iter]))+np.square(np.multiply(var_label[ID_iter],val_in[ID_iter])))).pdf(val_in[ID_iter]))

    pdfs = pdfs + np.finfo(np.float64).eps
    krig_likli = -np.log(10*pdfs).sum() # aviod numerical overflow
    krig_likli = krig_likli + n_in*np.log(10)
    return krig_likli

def cal_post_pdf(x,pos_in,val_in,covar_in,obs_label,prior_coeff,IDs,folder):
    "function ot calculate the posterior pdf for Laplace Approximation"
    """
        x is a list containing the parameters to be tuned, i.e., x[0] for tao, x[1] for sigma, x[2] for theta
            x[3] for reg_coeff, x[4] for for const, and x[5] for var_coeff, which identifies to level of obs error for crowd observations
        pos_in, and val_in are the positions and observed values of inputs
        covar_in are the observed values of covariates at pos_in
        prior_coeff is a dictionary containting all the variables required for prior pdf calculation
            prior_coeff['type'] is the type of prior, non_informative or informative
            prior_coeff['param'] is a dictionary containing the prior distribution parameters for the six elements in x
                if  prior_coeff['type']=='non_informative'
                    prior_coeff['param']['tao'] is the parameters for prior distribution of tao
                    prior_coeff['param']['sigma'] is the parameters for prior distribution of sigma
                    prior_coeff['param']['theta'] is the parameters for prior distribution of theta
                    prior_coeff['param']['reg_coeff'] is the parameters for prior distribution of reg_coeff
                    prior_coeff['param']['const'] is the parameters for prior distribution of const
                    prior_coeff['param']['var_coeff'] is the parameters for prior distribution of var_coeff
                if prior_coeff['type']=='informative'
                    prior_coeff['param']['mu'] is the means of the multinormal distribution
                    prior_coeff['param']['cov'] is the covariance matrix of the multinormal distribution
        """
    tao, sigma, theta =x.item(0), x.item(1), x.item(2)
    reg_coeff, const = x.item(3), x.item(4)
    var_coeff = x.item(5)

    n_in=pos_in.shape[0]
    var_label=var_coeff*obs_label
    krig_likli=cal_krig_likli_fast(pos_in, val_in, covar_in, var_label, tao, sigma, theta, reg_coeff, const, IDs, folder)

    try:
        tao_pdf = scipy.stats.uniform(loc=prior_coeff['param']['tao'][0], scale=prior_coeff['param']['tao'][1]).pdf(
            x[0])
        sigma_pdf = scipy.stats.uniform(loc=prior_coeff['param']['sigma'][0],
                                        scale=prior_coeff['param']['sigma'][1]).pdf(x[1])
        theta_pdf =  scipy.stats.uniform(loc=prior_coeff['param']['theta'][0], scale=prior_coeff['param']['theta'][1]).pdf(x[2])
        reg_coeff_pdf = scipy.stats.uniform(loc=prior_coeff['param']['reg_coeff'][0],
                                            scale=prior_coeff['param']['reg_coeff'][1]).pdf(x[3])
        const_pdf = scipy.stats.uniform(loc=prior_coeff['param']['const'][0],
                                        scale=prior_coeff['param']['const'][1]).pdf(x[4])
        var_coeff_pdf = scipy.stats.uniform(loc=prior_coeff['param']['var_coeff'][0],
                                            scale=prior_coeff['param']['var_coeff'][1]).pdf(x[5])

        prior_pdf = -np.log(tao_pdf) - np.log(sigma_pdf) - np.log(theta_pdf) - np.log(reg_coeff_pdf) - np.log(
            const_pdf) - np.log(var_coeff_pdf)
    except KeyError:
        prior_pdf = -np.log(scipy.stats.multivariate_normal.pdf(x, mean=prior_coeff['param']['mu'],
                                                                cov=prior_coeff['param']['cov']) + np.finfo(
            np.float64).eps)

    post_pdf=krig_likli+prior_pdf
    return post_pdf

def hessian ( x0, f, epsilon=1.e-5, linear_approx=False, *args ):
    """
    A numerical approximation to the Hessian matrix of cost function at
    location x0 (hopefully, the minimum)
    """
    # ``calculate_cost_function`` is the cost function implementation
    # The next line calculates an approximation to the first
    # derivative
    f1 = scipy.optimize.approx_fprime( x0, f, epsilon, *args)

    # This is a linear approximation. Obviously much more efficient
    # if cost function is linear
    if linear_approx:
        f1 = np.matrix(f1)
        return f1.transpose() * f1
    # Allocate space for the hessian
    n = x0.shape[0]
    hessian = np.zeros ( ( n, n ) )
    # The next loop fill in the matrix
    xx = x0
    for j in range( n ):
        xx0 = xx[j] # Store old value
        xx[j] = xx0 + epsilon # Perturb with finite difference
        # Recalculate the partial derivatives for this new point
        f2 = scipy.optimize.approx_fprime( xx, f, epsilon, *args)
        hessian[:, j] = (f2 - f1)/epsilon # scale...
        xx[j] = xx0 # Restore initial value of x0
    return hessian

def parallel_optimize(iterable,x0,pos_in,val_in,covar_in,obs_label,prior_coeff,IDs,folder):
    print ('+++++++++++++++++This is the '+str(iterable)+'th iteration+++++++++++++++')
    ini_per = 0.4 * np.random.rand(6, 1) + 0.8

    x_ini = np.multiply(x0, ini_per.ravel())
    bnds = ((0, 100), (0.001, 100), (10, None), (0, 2), (-100, 100), (0, 1))

    try:
        opt_x = scipy.optimize.minimize(cal_post_pdf, x_ini,
                                            args=(pos_in, val_in, covar_in, obs_label, prior_coeff, IDs, folder),
                                            method='L-BFGS-B',
                                            bounds=bnds)
        optimal_x_s = opt_x.x
        obj_s = cal_post_pdf(opt_x.x, pos_in, val_in, covar_in, obs_label, prior_coeff, IDs, folder)
    except ValueError:
        obj_s = np.nan
        optimal_x_s = np.zeros((6, 1))

    return [obj_s,optimal_x_s.ravel()]

def laplace_approx(pos_in,val_in,covar_in,obs_label,prior_coeff,folder):
    "function to calculate the laplace approximation and associated covariance matrix"
    """
        pos_in, and val_in are the positions and observed values of inputs
        covar_in are the observed values of covariates at pos_in
        prior_coeff is a dictionary containting all the variables required for prior pdf calculation
            prior_coeff['type'] is the type of prior, non_informative or informative
            prior_coeff['param'] is a dictionary containing the prior distribution parameters for the six elements in x
                if  prior_coeff['type']=='non_informative'
                    prior_coeff['param']['tao'] is the parameters for prior distribution of tao
                    prior_coeff['param']['sigma'] is the parameters for prior distribution of sigma
                    prior_coeff['param']['theta'] is the parameters for prior distribution of theta
                    prior_coeff['param']['reg_coeff'] is the parameters for prior distribution of reg_coeff
                    prior_coeff['param']['const'] is the parameters for prior distribution of const
                    prior_coeff['param']['var_coeff'] is the parameters for prior distribution of var_coeff
                if prior_coeff['type']=='informative'
                    prior_coeff['param']['mu'] is the means of the multinormal distribution
                    prior_coeff['param']['cov'] is the covariance matrix of the multinormal distribution
        folder is the working folder where the .csv files are writing in
        """
    n_in = pos_in.shape[0]
    IDs = np.random.permutation(n_in)  # the indices for random selection in the later stage

    reg_stat=scipy.stats.linregress(np.asarray(covar_in).T,np.asarray(val_in).T)
    reg_coeff, const = reg_stat[0], reg_stat[1]
    err_in = val_in - reg_coeff * covar_in - const
    err_export=np.column_stack((pos_in, err_in))
    err_export = pd.DataFrame(err_export, columns=['x', 'y', 'obs'])
    file_name_gauge = folder + 'bayes_krig.csv'
    err_export.to_csv(file_name_gauge, index=False)
    os.system('"Rscript.exe" --nosave bayes_krig.R')
    variogram_coeff = pd.read_csv(folder + 'variogram_coeff.csv', sep=',', usecols=[1])
    variogram_coeff=variogram_coeff.as_matrix()
    tao=np.sqrt(variogram_coeff.item(0))

    sigma=np.sqrt(variogram_coeff.item(1))
    theta=np.abs(variogram_coeff.item(2))
    var_coeff=0.1
    x0=np.asarray([tao,sigma,theta,reg_coeff,const,var_coeff])

    N_repeat=3

    results = Parallel(n_jobs=N_repeat)(delayed(parallel_optimize)(iterable,x0,pos_in,val_in,covar_in,obs_label,prior_coeff,IDs,folder)\
                                        for iterable in range(N_repeat))
    results_best = min(results,key=lambda t:t[0])
    if np.isnan(results_best[0]):
        print ('++++++++++++++++++++CAUTION: No reasonable optimization+++++++++++++++++++')
        optimal_x = x0
    else:
        optimal_x=results_best[1]

    hessian_x=hessian(optimal_x,cal_post_pdf,1.e-5,False,pos_in,val_in,covar_in,obs_label,prior_coeff,IDs,folder)
    cov=np.linalg.pinv(hessian_x)
    cov=nearSPD_R(cov,folder)
    return optimal_x, cov

def bayes_krig_deter(x,pos_in,pos_out,val_in,covar_in,covar_out,folder):
    "function to calculate the deterministic prediction of bayes kriging"
    """
        x is a optimal list containing the parameters to be tuned, i.e., x[0] for tao, x[1] for sigma, x[2] for theta
            x[3] for reg_coeff, x[4] for for const, and x[5] for var_coeff, which identifies to level of obs error for crowd observations
        pos_in, and val_in are the positions and observed values of inputs
        pos_out are the positions of the outputs to be estimated
        covar_in are the observed values of covariates at pos_in
        covar_out are the observed values of covarites at pos_out
        """
    tao, sigma, theta = x.item(0), x.item(1), x.item(2)
    reg_coeff, const = x.item(3), x.item(4)

    err_in = val_in - reg_coeff * covar_in - const
    krig_esti, krig_sigma = cal_krig_esti_sk(pos_out, pos_in, err_in, tao, sigma,theta,folder)
    krig_esti=krig_esti.flatten()
    krig_esti = reg_coeff * covar_out + krig_esti + const
    krig_esti=krig_esti.T
    krig_sigma = krig_sigma.flatten().T
    krig_esti = pd.DataFrame(krig_esti)
    krig_sigma = pd.DataFrame(krig_sigma)
    krig_esti, krig_esti_0, krig_sigma = rain_field_1d_to_2d(krig_esti, krig_sigma, 0)
    return krig_esti_0, krig_sigma

def bayes_krig_stocha_parallel(iterable,x,cov,val_in,covar_in,pos_out,pos_in,folder,covar_out):
    # cov = np.multiply(cov, cov > 0)
    while True:
        try:
            xi = np.random.multivariate_normal(x, cov)
            tao, sigma, theta = xi.item(0), xi.item(1), xi.item(2)
            reg_coeff, const = xi.item(3), xi.item(4)

            err_in = val_in - reg_coeff * covar_in - const
            krig_esti, krig_sigma = cal_krig_esti_sk(pos_out, pos_in, err_in, tao, sigma, theta, folder)
            krig_esti = krig_esti.flatten()
            krig_esti = reg_coeff * covar_out + krig_esti + const
            krig_esti = krig_esti.T
            krig_sigma = krig_sigma.flatten().T
            krig_esti = pd.DataFrame(krig_esti)
            krig_sigma = pd.DataFrame(krig_sigma)
            krig_esti, krig_esti_0, krig_sigma = rain_field_1d_to_2d(krig_esti, krig_sigma, 0)
            return krig_esti_0
            break
        except ValueError:
            print ('+++++++++++++++++++Opps, unreasonable parameters for kriging, try again++++++++++++++++++')

def bayes_krig_stocha(x,cov,pos_in,pos_out,val_in,covar_in,covar_out,N_D,N_realization,folder):
    "function to calculate the stochastic prediction of bayes kriging"
    """
        x is a optimal list containing the parameters to be tuned, i.e., x[0] for tao, x[1] for sigma, x[2] for theta
            x[3] for reg_coeff, x[4] for for const, and x[5] for var_coeff, which identifies to level of obs error for crowd observations
        cov is the associated covariance matrix of x according to laplaces approximation
        pos_in, and val_in are the positions and observed values of inputs
        pos_out are the positions of the outputs to be estimated
        covar_in are the observed values of covariates at pos_in
        covar_out are the observed values of covarites at pos_out
        N_D is a arrary of the dimensions of the output, e.g. [200,100]
        N_realization is the number of realizations for the outputing rainfall fields
        """
    bayes_krig_out=np.zeros((N_D[0],N_D[1],N_realization))
    # try
    results = Parallel(n_jobs=4)(
        delayed(bayes_krig_stocha_parallel)(iterable,x,cov,val_in,covar_in,pos_out,pos_in,folder,covar_out) \
        for iterable in range(N_realization))

    for i in range(N_realization):
        bayes_krig_out[:,:,i]=results[i]

    return bayes_krig_out

def gen_bayes_krig_stocha_data(prior_coeff,folder,N_D,N_realization):
    "function to generate stochastic merged rainfall field based on the bayes kriging method"
    """
        prior_coeff is a dictionary containting all the variables required for prior pdf calculation
            prior_coeff['type'] is the type of prior, non_informative or informative
            prior_coeff['param'] is a dictionary containing the prior distribution parameters for the six elements in x
                if  prior_coeff['type']=='non_informative'
                    prior_coeff['param']['tao'] is the parameters for prior distribution of tao
                    prior_coeff['param']['sigma'] is the parameters for prior distribution of sigma
                    prior_coeff['param']['theta'] is the parameters for prior distribution of theta
                    prior_coeff['param']['reg_coeff'] is the parameters for prior distribution of reg_coeff
                    prior_coeff['param']['const'] is the parameters for prior distribution of const
                    prior_coeff['param']['var_coeff'] is the parameters for prior distribution of var_coeff
                if prior_coeff['type']=='informative'
                    prior_coeff['param']['mu'] is the means of the multinormal distribution
                    prior_coeff['param']['cov'] is the covariance matrix of the multinormal distribution
         folder is the working folder where the .csv files are writing in
        """
    gauge_data=pd.read_csv(folder + 'gauge_rainfall.csv', sep=',', usecols=range(0,4))
    crowd_data=pd.read_csv(folder + 'crowd_rainfall.csv', sep=',', usecols=range(0,4))
    radar_data = pd.read_csv(folder + 'radar_rainfall.csv', sep=',', usecols=range(0, 3))
    radar_data = np.asarray(radar_data)
    gauge_data=np.asarray(gauge_data)
    crowd_data=np.asarray(crowd_data)
    gauge_data=np.concatenate((gauge_data, np.zeros((gauge_data.shape[0], 1))), axis=1)
    crowd_data=np.concatenate((crowd_data, np.ones((crowd_data.shape[0], 1))), axis=1)
    all_data=np.concatenate((gauge_data,crowd_data),axis=0)
    pos_in=all_data[:,0:2]
    obs_label=all_data[:,4]
    pos_out = radar_data[:,0:2]

    val_in = all_data[:, 2]
    covar_in = all_data[:, 3]
    covar_out = radar_data[:, 2]

    optimal_x,cov = laplace_approx(pos_in, val_in, covar_in, obs_label, prior_coeff, folder)
    krig_esti, krig_sigma = bayes_krig_deter(optimal_x, pos_in, pos_out, val_in, covar_in, covar_out, folder)
    bayes_krig_out = bayes_krig_stocha(optimal_x,cov,pos_in,pos_out,val_in,covar_in,covar_out,N_D,N_realization,folder)

    return bayes_krig_out, krig_esti, krig_sigma

if __name__ == '__main__':
    folder='***' # enter the working folder
    prior_param = {'tao': [0, 200], 'sigma': [0, 200], 'theta': [0, 200], 'reg_coeff': [-2, 4],
                   'const': [-20, 40], 'var_coeff': [0, 1]}
    prior_coeff = {'type': 'non_informative', 'param': prior_param}
    N_D = [200, 100]
    n_realization=100
    bayes_krig_out, krig_esti, krig_sigma = gen_bayes_krig_stocha_data(prior_coeff, folder, N_D, n_realization)
    plt.imshow(krig_esti)
    plt.show()




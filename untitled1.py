PRUNE_GAMMA = 0.0001      # threshold for prunning small gamma_i
p           = 0.8         # p-norm
EPSILON     = 1e-8        # threshold for stopping iteration. 
MAX_ITERS   = 800         # maximum iterations
PRINT       = 0           # not show progress information
import numpy.matlib


def MFOCUSS(Phi, Y, lambda_):
    N,M = Phi.shape 
    N,L= Y.shape
    gamma = np.ones(M);         # initialization of gamma_i
    mu = np.zeros((M,L));           # initialization of the solution matrix
    count = 0;                 # record iterations

    for i in range(200):
        '''
        % =========== Prune weights as their hyperparameters go to zero ===========
        if (min(gamma) < PRUNE_GAMMA )
            index = find(gamma > PRUNE_GAMMA);
            gamma = gamma(index);   
            Phi = Phi(:,index);            % corresponding columns in Phi
            keep_list = keep_list(index);
            m = length(gamma);
    
            if (m == 0)   break;  end;
        end;
        '''

        # ====== Compute new weights ======
        G = np.matlib.repmat(np.sqrt(gamma),N,1)
        PhiG = np.multiply(Phi,G)
        U,S,V = np.linalg.svd(PhiG,full_matrices=False)    
        d1= S.shape[0]
        if (d1 > 1):
            diag_S = np.diag(S)
        else:         
            diag_S = S[0]
        vec1 = np.divide(diag_S,(diag_S**2 + np.sqrt(lambda_) + 1e-16)).T
        #vec =  np.matlib.repmat(vec1,N,1)

        U_scaled = np.multiply(U[:,0:np.min((N,M))],vec1)
        Xi = np.multiply(G.T,(V.T @ U_scaled.T));
    
        mu_old = mu;
        mu = Xi @ Y;
    
    
        # *** Update hyperparameters ***
        gamma_old = gamma;
        mu2_bar = np.sum(abs(mu)**2,1);
        gamma = (mu2_bar/L)**(1-p/2);
        '''
        % ========= Check stopping conditions, etc. ========= 
        count = count + 1;
        if (PRINT) disp(['iters: ',num2str(count),'   num coeffs: ',num2str(m), ...
                '   gamma change: ',num2str(max(abs(gamma - gamma_old)))]); end;
        if (count >= MAX_ITERS) break;  end;
    
        if (size(mu) == size(mu_old))
            dmu = max(max(abs(mu_old - mu)));
            if (dmu < EPSILON)  break;  end;
        end;
    
    end;
    
    
    gamma_ind = sort(keep_list);
    gamma_est = zeros(M,1);
    gamma_est(keep_list,1) = gamma;  
        '''
    return mu
    
#%%
U = MFOCUSS(gain,whitened_data,0.01)
#%%
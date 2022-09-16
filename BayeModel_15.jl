import Pkg
using Distributions
using BenchmarkTools
using Base.Threads: @spawn, @threads
using CSV
using DataFrames
using Random

function likelihoodeach(t1::Float64, t_star::Float64, mu::Float64)
    # this functions aims to calculate the likelihood under each specific tb(tb[i]) and t_star(t_star[j])
   
    t2 = t1+2*t_star   
    P = 3/4-3/4*exp(-mu*t2)

    return P
end


function thetaUpdate(theta::Float64, JumpingWidth::Float64, theta_lb::Float64=0.0)

    theta_new = rand(Uniform(max(theta_lb, theta-JumpingWidth), theta+JumpingWidth))
    return(theta_new)
end


function tbUpdate(tb::Float64, tb_lb::Float64, tb_ub::Float64, JumpingWidth::Float64)

    tb_new = rand(Uniform(max(tb_lb, tb-JumpingWidth), min(tb+JumpingWidth,tb_ub)))
    return(tb_new)
end


function TransNet(input::Matrix{Float64}, D::Matrix{Float64}, mu::Float64, N::Float64, inf_rate::Float64, Jump_theta::Float64,
    itr_t_star::Int64=50000, itr_MCMC::Int64=100000, Pevent=0, burn_in::Int64=5000)
    # This function is aiming to find the most possible transmission network using temporal information: input, which is a n*3 matrix containing the 
    # individual ID, symptom on set time, and removal time of n infected individuals. D is a n*n matrix including the SNP data between each pair of
    # infects. mu is the mutation rate (per site per day) of Tuberculosis. N is the number of pathogen base pairs. itr_t_star is the number of iterations
    # that used to compute the MCMC integral of t_star. itr_MCMC is the number of iterationsof MCMC. burnin is the number of iterations that we discard 
    # (default is the first 5000 iterations).
    
    n = size(input, 1)
    # initialize 
    if Pevent == 0
        Pevent = zeros(Float64, n, n)
        for i in 2:n
            for j in 1:(i-1)
                Pevent[j,i] = log(1.0/float(i-1))
            end
        end
    end
    if size(input, 2) != 3
        return "Input data has incorrect format: there should be three columns containing individual ID, symptom onset time and removal time"
    elseif (n != length(unique(input[:,1])))
        return "incorrect ID labeling of infectious individuals (duplicates or incontinuous)"
    elseif size(Pevent, 1) != n | size(Pevent, 2) != n
        return "incorrect assignment of Pevent (Matrix of Prior Probability of transmission event )"
    else
        # initialize the parameters
        num_out = Int64(floor((itr_MCMC - burn_in)/100)) #vector/list length of each output
        Net_infIDout = zeros(Int64, num_out, n) #output for the most likely individual who infected individual i (i is the sequence/row number)
        Param_out = zeros(Float64, num_out, 2) #output to store loglikelihood & theta
        Tb_out = zeros(Float64, n, n, num_out) #output of tb (in matrix) corresponding to each plausible transmission event.
        ProbNet_event_out = zeros(Float64, num_out, n) #output of ProbNet_event (in every 250 iterations)
        #theta_out = zeros(Float64, num_out) #output of theta (in every 250 iterations).

        ProbNet_event = zeros(Float64, n) #the transmission probability of the most likely transmission event that each individual got infected.
        Net_infID = zeros(Int64, n) #the most likely individual who infected individual i (i is the sequence/row number)
        #Fact_D = zeros(Float64, n, n) #matrix to store the coefficients based on SNP data between each pairs of individulas
        TAB_Mtx= zeros(Float64, n, n) #matrix to store whether the symptom onset time time difference between each pairs of individuals.
        TRATIB_Mtx = zeros(Float64, n, n)
        Tb_sim = zeros(Float64, n, n) #matrix to store the tb values (latent period) of whom got infected in the corresponding transmission event.
        Tb_Proposal_Dis = zeros(Float64, n, n) #matrix to store the probability of proposal distribution of tb (truncated chi-square) corresponding 
        # to each plausible transmission event.
        T1_Mtx = zeros(Float64, n, n) #matrix to store t1 corresponding to each plausible transmission event
        Jump_tb_Mtx = zeros(Float64, n, n)
        Tb_lb = zeros(Float64, n, n)
        Prob_prod = zeros(Float64, n, n)
        TRA_TIB_TIA_min =  zeros(Float64, n, n)
        scal = 2/28 #2/28 is within 180 - 270 days (0.5 - 0.75 years). In other words, the expectation of latent period lies within 6 - 9 months
        
        Pro_theta_new = 0.0
        Pro_Tb = zeros(Float64, n, n) #matrix to store the pdf of given tb corresponding to each plausible transmission event.
        Pro_Phi_Mtx = zeros(Float64, n, n) #matrix to store the probability of tranmission network. 
        t_star_sim_new = Vector{Float64}(undef,itr_t_star) #vector to store the newly simulated t_star
        theta_sim_new = 0.0
       
        # simulate initial value of theta
        theta_sim = rand(Exponential(1))
        theta_lb = 0.1 # lower bound of theta
        theta_ub = 1.5 # upper bound of theta
        # making sure that our starting point is not too biased
        while theta_sim < theta_lb || theta_sim > theta_ub
            theta_sim = rand(Exponential(1))
        end
        theta_sim = 1.0
        Pro_theta = pdf(Exponential(1), theta_sim) #pdf of given theta
        t_star_sim = rand(Exponential(theta_sim), itr_t_star) #theta = 2*Ne*mu

        # calculate the Likelihood at starting point. Calculate and store Fact_D & InfTime_diff at the same time.
        # Consider individual j was infected by individual i.
        Tb_sim[1,1] = 0.57 # mean of expected latent period.
        Net_infID[1] = 1

        for j in 2:n
            TIB = input[j,2] #symptom on set time of individual j
            TRB = input[j,3] #removal time of indiviudal j
            SNP = 1000

            for i in 1:(j-1) 
                TIA = input[i,2] #symptom on set time of individual i
                #Fact_D[i,j] = factorialSNP(D[i,j],N) #Calculate the coefficients of I1 and I2 based on SNP data
                TRA = input[i,3] #removal time of individual i
                TAB_Mtx[i,j] = TIB-TIA
                TRATIB_Mtx[i,j] = TIB-TRA

                if TRATIB_Mtx[i,j] <= 2 && TAB_Mtx[i,j] > 0
                    #calculate the maximum latent period that individual B could have
                    tAB = TAB_Mtx[i,j] #the upper limit of tb and t_star
                    TRA_TIB_TIA_min[i,j] = min(TRA-TIA, tAB)
                    
                    #fact = Fact_D[i,j]
                    d = D[i,j]

                    if d < SNP
                        SNP = d
                        Net_infID[j] = i
                    end

                    Tb_lb[i,j] = max(0,TIB-TRA)
                    tb_l = Tb_lb[i,j]#the lower limit of tb
                    Jump_tb_Mtx[i,j] = (tAB-tb_l)/5.0

                    Tb_sim[i,j] = tb_l+(tAB-tb_l)*rand()
                    tb = Tb_sim[i,j]
                    
                    Tb_Proposal_Dis[i,j] = cdf(Chisq(8), tAB/scal)-cdf(Chisq(8), tb_l/scal)
                    Pro_Tb[i,j] = pdf(Chisq(8), tb/scal)
            
                    Pro_Phi_Mtx[i,j] = (tb-tAB)/inf_rate+log(1-exp(-(TRA_TIB_TIA_min[i,j]+tb-tAB)/inf_rate))
                    Pro_Phi = Pro_Phi_Mtx[i,j]

                    T1_Mtx[i,j] = TRA+TRB+2*tb-2*TIB
                    t1 = T1_Mtx[i,j]

                    temp = 0.0

                    if Pro_Phi != -Inf 
                         
                        for l in 1:itr_t_star
                            t_star = t_star_sim[l]
                            temp+= ifelse(t_star >= tAB, 0, likelihoodeach(t1::Float64, t_star::Float64, mu::Float64))
                        end
                        P = temp/itr_t_star
                        logLike = d*log(P)+(N-d)*log(1-P)
                        
                        Prob_prod[i,j] = logLike+Pro_Phi+Pevent[i,j]+log(Pro_Tb[i,j])+log(Tb_Proposal_Dis[i,j])                      
                    end                                    
                end                
            end
            #sum_j_col = sum(Prob_prod[:,j]) # prbability sum of the transmission event that individual j got infected. 
            # if the sum equal to 0, it means we haven't found any suspicious infected individual that infect individual j.
            # In other words, individual j is infected by someone else outside the transmission network based on current parameters. 
            
            ProbNet_event[j] = Prob_prod[Net_infID[j],j]
        end

        for k in 2:itr_MCMC
            
            #nonzero_index = (ProbNet_event .!= 0) #get the index of individual possess non-zeros transmission probability
            #Prob_net_nonzero = ProbNet_event[nonzero_index] #get the maximum non-zeros probability of transmission event of each individual.
            H = 0.0 # store hasting ratio
            
            if rand() <= 0.1
                ## update theta
                theta_sim_new = thetaUpdate(theta_sim, Jump_theta, 0.0)
                t_star_sim_new = rand(Exponential(theta_sim_new),itr_t_star)
                ProbNet_event_new = zeros(Float64, n)

                for j in 2:n
                    i = Int64(Net_infID[j]) # individual i infected individual j in the last loop
                    if ProbNet_event[j] != 0
                        tAB = TAB_Mtx[i,j]
                        Pro_theta_new = pdf(Exponential(1), theta_sim_new)
                        t1 = T1_Mtx[i,j]
                        #fact = Fact_D[i,j]
                        d = D[i,j] 

                        temp = 0.0
                        P = 0.0
                         
                        for l in 1:itr_t_star
                            t_star = t_star_sim_new[l]
                            temp+= ifelse(t_star >= tAB, 0, likelihoodeach(t1::Float64, t_star::Float64, mu::Float64))
                        end
                        P = temp/itr_t_star
                        logLike = d*log(P)+(N-d)*log(1-P)

                        ProbNet_event_new[j] = logLike+Pro_Phi_Mtx[i,j]+Pevent[i,j]+log(Pro_Tb[i,j])+log(Tb_Proposal_Dis[i,j])

                    end
                end
                #Prob_net_nonzero_new = ProbNet_event_new[nonzero_index]
                H = min(1, exp(sum(ProbNet_event_new)+log(Pro_theta_new)-sum(ProbNet_event)-log(Pro_theta)))
                if H > rand()
                    theta_sim = theta_sim_new
                    ProbNet_event = ProbNet_event_new 
                    t_star_sim = t_star_sim_new
                    Pro_theta = Pro_theta_new 
                    #Prob_net_nonzero = Prob_net_nonzero_new
                end

            else
                ## update one transmission event and its corresponding tb
                # now randomly change one transmission event
                m = rand(2:n)
                # randomly select one individual (before m) that infected m, making sure the individual that infected m is not the 
                # the one in the previous network.
                
                ID_pool = shuffle(1:(m-1))
                
                for q in 1:(m-1)
                    infID = ID_pool[q]
                    if (TRATIB_Mtx[infID,m] <= 2 && TAB_Mtx[infID,m] > 0)
                        ProbeventID_new = 0.0 
                        tAB = TAB_Mtx[infID,m]
                
                        tb = Tb_sim[infID,m]
                        tb_l = Tb_lb[infID,m]
                        jump_tb = Jump_tb_Mtx[infID,m]
                        tb_new = tbUpdate(tb, tb_l, tAB, jump_tb)
                        Pro_Tb_new = pdf(Chisq(8), tb_new/scal)
                        Pro_Phi_new = (tb_new-tAB)/inf_rate+log(1-exp(-(TRA_TIB_TIA_min[infID,m]+tb_new-tAB)/inf_rate))
                        t1_new = T1_Mtx[infID,m]-2*tb+2*tb_new
                        #fact = Fact_D[infID,m]
                        d = D[infID,m]

                        temp = 0.0

                        for l in 1:itr_t_star
                            t_star = t_star_sim[l]
                            temp+= ifelse(t_star >= tAB, 0, likelihoodeach(t1_new::Float64, t_star::Float64, mu::Float64))
                        end
                        P = temp/itr_t_star
                        logLike = d*log(P)+(N-d)*log(1-P)
                    
                        ProbeventID_new = logLike+Pro_Phi_new+Pevent[infID,m]+log(Pro_Tb_new)+log(Tb_Proposal_Dis[infID,m])
                    
                        ProbeventID = ProbNet_event[m]
                
                        if min(1,exp(ProbeventID_new-Prob_prod[infID,m])) > rand()
                            T1_Mtx[infID,m] = t1_new
                            Tb_sim[infID,m] = tb_new
                            Pro_Phi_Mtx[infID,m] = Pro_Phi_new
                            Pro_Tb[infID,m] = Pro_Tb_new
                            Prob_prod[infID,m] = ProbeventID_new
                        end
                
                        H = min(1, exp(ProbeventID_new-ProbeventID))
                        if H > rand() 
                            ProbNet_event[m] = ProbeventID_new
                            Net_infID[m] = infID
                            #nonzero_index = ifelse(ProbeventID == 0, (ProbNet_event .!= 0), nonzero_index)
                        end            
                    end
                end
            end  
                
                
                
                        
            if k > burn_in && k%100 == 0
                h = Int64(ceil((k-burn_in)/100))
                Net_infIDout[h,:] = Net_infID
                Tb_out[:,:,h] = Tb_sim
                ProbNet_event_out[h,:] = ProbNet_event
                Param_out[h,1] = sum(ProbNet_event)+log(Pro_theta)
                Param_out[h,2] = theta_sim
            end
        end
        return Net_infIDout, Tb_out, ProbNet_event_out, Param_out
    end
end

testDDD = Array(DataFrame(CSV.File("/Users/huhuimin/Documents/STAT/Research/Code/Real Data Study/Input/time_real_data.csv")))
testDD = testDDD[:,3:5]
testD = zeros(Float64,length(testDDD[:,1]),3)
testD[:,2:3] = float(testDD[:,2:3])/365
testD[:,1] = float(testDD[:,1])
DDD = Array(DataFrame(CSV.File("/Users/huhuimin/Documents/STAT/Research/Code/Real Data Study/Input/SNP_real_data.csv")))
DD = DDD[:,2:70]
D = float(DD)

# @time ppp = TransNet(testD, D, 5.5e-10, 4e+06, 0.02, 500, 0)
@time ppp = TransNet(testD, D, 2.4e-7, 4411532.0, 7.3, 0.05, 50000, 100000, 0, 0) #new1
#@time pp = TransNet(testD, D, 2.4e-7, 4411532.0, 11.0, 0.003, 50000, 500000, 0, 0) #new2
#@time p = TransNet(testD, D, 2.4e-6, 4411532.0, 7.3, 0.003, 50000, 500000, 0, 0) #new3

inf_Net_ppp = DataFrame(ppp[1],:auto)
Param_ppp = DataFrame(ppp[4],:auto)
Tb_ppp = ppp[2]
#inf_Net_pp = DataFrame(pp[1],:auto)
#Param_pp = DataFrame(pp[4],:auto)
#Tb_pp = pp[2]
#inf_Net_p = DataFrame(p[1],:auto)
#Param_p = DataFrame(p[4],:auto)
#Tb_p = p[2]
# Tb_ppp = ppp[2] 
# Tb_pp = pp[2]
# Tb_sam_ppp = DataFrame(Tb_ppp[9,10])
# Tb_sam_pp = DataFrame(Tb_pp[9,10], Tb_pp[19,20])

CSV.write("/Users/huhuimin/Documents/STAT/Research/Code/Real Data Study/Output/test/inf_NetID_new1.csv", inf_Net_ppp)
CSV.write("/Users/huhuimin/Documents/STAT/Research/Code/Real Data Study/Output/test/Para_Net_new1.csv", Param_ppp)
#CSV.write("/Users/huhuimin/Documents/STAT/Research/Code/Real Data Study/Output/new/inf_NetID_new2.csv", inf_Net_pp)
#CSV.write("/Users/huhuimin/Documents/STAT/Research/Code/Real Data Study/Output/new/Para_Net_new2.csv", Param_pp)

#CSV.write("/Users/huhuimin/Documents/STAT/Research/Code/Real Data Study/Output/new/inf_NetID_new3.csv", inf_Net_p)
#CSV.write("/Users/huhuimin/Documents/STAT/Research/Code/Real Data Study/Output/new/Para_Net_new3.csv", Param_p)
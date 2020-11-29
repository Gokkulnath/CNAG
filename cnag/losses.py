import torch


def fooling_objective(qc_):
    '''Helper function to computer compute -log(1-qc'), 
    where qc' is the adversarial probability of the class having 
    maximum probability in the corresponding clean probability
    qc' ---> qc_
    Parameters: 
    prob_vec : Probability vector for the clean batch
    adv_prob_vec : Probability vecotr of the adversarial batch
    Returns: 
    -log(1-qc') , qc'
    
    '''  
    # Get the largest probablities from predictions : Shape (bs,1)
    qc_=qc_.mean()
    return -1*torch.log(1-qc_) , qc_

def diversity_objective(prob_vec_no_shuffle, prob_vec_shuffled):
    '''Helper function to calculate the cosine distance between two probability vectors
    Parameters: 
    prob_vec : Probability vector for the clean batch
    adv_prob_vec : Probability vector for the adversarial batch
    Returns : 
    Cosine distance between the corresponding clean and adversarial batches
    '''    
    return torch.cosine_similarity(prob_vec_no_shuffle,prob_vec_shuffled).mean()

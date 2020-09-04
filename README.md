# Statistical_Inference_for_Hawkes_Processes_with_Deep_Learning
 
This is the code repository for my Master Thesis which was submitted in partial fulfilment of the requirements for the MSc in Statistics of Imperial College London.

A description of each file is presented here:


Comparison_of_methods: This file contains the code which was used to run the comparison of methods test seen in Figure 5.12. This code will not run without errors, as the required data for the MC-EM algorithm results is being withheld at the request of the author. This data will be added to this repository once their paper has passed peer review.

dueling_decoders: This file contains the code which was used to train and test the dueling decoder framework. This file is responible for the results in Figure 5.3 - 5.6

Execute_tests: This file executes Test_of_effect_of_activity, Test_of_effect_of_beta, Test_of_effect_of_delta, Test_of_effect_of_eta, and Comparison_of_methods.

Hawkes Process Example: This code generated the example plot seen in Figure 2.2

HP_Discretise: This code provides the user defined functions regarding the conversion of a Hawkes process to an aggregated Hawkes process.

MLE: This code contains the required MLE functions and the function for calculating the true intensity.

Neg_bin_VAE: This file contains the code which was used to trian and test the negative binomial VAE.

Poisson Process Example: This code generated the example plot seen in Figure 2.1

Poisson_VAE: This file contains the code which was used to train and test the dueling decoder framework. This file is responible for the results in Figure 5.1 and 5.2

Stan_test: This file is a stan file and contains the stan model specification which is called by the Poisson VAE, Dueling Decoders and Negative Binomial VAE

Supervised_method: This file contains the code for the supervised learning method outlined in Section 4.3

Test_of_effect_of_activity: This code generates the boxplots seen in Figure 5.11 

Test_of_effect_of_beta: This code generates the boxplots seen in Figure 5.8 

Test_of_effect_of_delta: This code generates the boxplots seen in Figure 5.9

Test_of_effect_of_eta: This code generates the boxplots seen in Figure 5.10 


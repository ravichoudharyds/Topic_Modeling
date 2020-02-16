# Topic Modeling using LDA, VAE and GAN

In this project, I compared topics derived from an Online LDA model implemented using gensim, Autoencoding variational inference for topic models (AVITM) introduced by Akash Srivastava and Charles Sutton and Adversarial-neural topic model (ATM) introduced by Rui Wang, Deyu Zhou, and Yulan He.

[AVITM](https://arxiv.org/abs/1703.01488) model was taken from @hyqneuron [github repo](https://github.com/hyqneuron/pytorch-avitm). I have implemented ATM based on the details mentioned in the [paper](https://arxiv.org/abs/1811.00265). All of the deep learning models were implemented in PyTorch.

AVITM model uses VAE for inference, therefore it uses an approximation of Laplace distribution with logistic-normal distribution. While ATM paper uses GAN for inference. I evaluated the model against each other using the topic coherence scores from gensim package. My results from running just one iteration showed that AVITM and ATM can outperform traditional LDA by giving more meaningful and independent topics. I would suggest to repeat the experiments a few more times to get a range of scores and to draw conclusions on the perormance of the models.

## Caveat:
The topic coherence unfortunately couldn't be calculated using wiki corpus and was instead calculated from the documents themselves. Also early stopping was used for selecting the best ATM/AVITM models using the Topic Coherence score.  

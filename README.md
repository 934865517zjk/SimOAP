## SimOAP: Improve Coherence and Consistency in Persona-based Dialogue Generation via Over-sampling and Post-evaluation ##
The code of SimOAP in the ACL2023 main conference.
Paper link: [link](https://aclanthology.org/2023.acl-long.553/)

## First Step: Over-sampling ##
We used BERT-over-BERT (BoB) and Multi-GPT2 as backbone models, so you need to first generate large-scale candidate responses based on their codes. In our setting, `n=2000` candidate responses are generated for each history utterances.
And top-k sampling is used, where k is set to 100 (`k=100`).

The code links for the two backbone models are as follows:

BERT-over-BERT (BoB): https://github.com/songhaoyu/BoB 

> Implementation details: BoB has two decoders, the first decoder is used to generate a preliminary response and the second decoder is used to modify the preliminary response and generate the final response. We only use top-k sampling in the first decoder. The second decoder is a response modifier, so we use greedy search. 

Multi-GPT2: https://github.com/caoyu-noob/Multi-GPT2

> Implementation details: we directly use top-k sampling for sampling. 

## Second Step: Post-evaluation ##
Post-evaluation consists of two parts: coherence evaluation and consistency evaluation. In coherence evaluation, the TF-IDF algorithm is used. In consistency evaluation, natural language inference (NLI) is used.

TF-IDF algorithm

    python coherence_evaluation.py

Natural Language Inference
    
    python consistent_evaluation.py

For the weights of the NLI model, you need to download it from the following link and place it into the consistent_model/ folder: [link](https://drive.google.com/file/d/1MX-V1mdGwfZ4Rqhjo87XEZtN2Snzw9M_/view?usp=drive_link)

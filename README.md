![contrast](https://user-images.githubusercontent.com/40825205/188257392-ecb6965a-57c7-42bf-8fa1-f533fb403fb1.png)


# Abstract
In recent years, open-domain question answering (OpenQA) draws significant research attention due to its wide range of applications. However, one concern for the existing OpenQA system is its slow inference speed for being used in practical scenarios. Aiming at this concern, the idea of indexing possible asked questions (referred to as PAQ) is proposed. The idea is two-fold: (1) pre-generating and indexing all possible asked questions and the corresponding answers and (2) processing on-line queries by retrieving similar QA pairs through index structures. In this paper, we extend the PAQ idea by considering \textit{QA pair merging} to boost the accuracy of the PAQ retriever. We find that embedding merged questions (rather than a single question) leads to better representation that improves the accuracy of the QA pair retriever. Along with the question merging idea, we propose MPAQ Retriever, a retriever for merged QA pairs, based on \textit{self-filtering} and \textit{self-supervised contrastive learning} techniques. The performance evaluation demonstrates an improvement of 7.1\% (1\%) on TriviaQA (Natural Question).

for the original data of PAQ, please refer to https://github.com/facebookresearch/PAQ

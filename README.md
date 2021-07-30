# GNN-NIDS

Link to paper: [Unveiling the potential of Graph Neural Networks for robust Intrusion Detection]()

David Pujol-Perich, José Suárez-Varela, Albert Cabellos-Aparicio, Pere Barlet-Ros

Contact us: *contactus@bnn.upc.edu*

## Abstract
The last few years have seen an increasing wave of attacks with serious economic and privacy damages, which evinces the need for accurate Network Intrusion Detection Systems (NIDS). Recent works propose the use of Machine Learning (ML) techniques for building such systems (e.g., decision trees, neural networks). However, existing ML-based NIDS are barely robust to common adversarial attacks, which limits their applicability to real networks. A fundamental problem of these solutions is that they treat and classify flows independently. In contrast, in this paper we argue the importance of focusing on the structural patterns of attacks, by capturing not only the individual flow features, but also the relations between different flows (e.g., the source/destination hosts they share). To this end, we use a graph representation that keeps flow records and their relationships, and propose a novel Graph Neural Network (GNN) model tailored to process and learn from such graph-structured information. In our evaluation, we first show that the proposed GNN model achieves state-of-the-art results in the well-known CIC-IDS2017 dataset. Moreover, we assess the robustness of our solution under two common adversarial attacks, that intentionally modify the packet size and inter-arrival times to avoid detection. The results show that our model is able to maintain the same level of accuracy as in previous experiments, while state-of-the-art ML techniques degrade up to 50% their accuracy (F1-score) under these attacks. This unprecedented level of robustness is mainly induced by the capability of our GNN model to learn flow patterns of attacks structured as graphs.

## Description
This repository includes the implementations presented in **Unveiling the potential of Graph Neural Networks for robust Intrusion Detection**.

In this paper we present a novel Network Intrusion Detection System (NIDS) based on Graph Neural Networks that not only proves to obtain state-of-the-art results when evaluated with a well-known IDS dataset, but also proves to be extremely robust to common evation techniques (i.e., adversarial attacks).

<p align="center"> 
  <img src="/assets/overview.png" width="700" alt>
</p>

In this regard, this repository includes the fully functional model that we used for this paper, which is implemented using IGNNITION, a novel framework that allows fast prototyping of GNN models. In addition to this, we also include a naitve implementation of the model based on TensorFlow. (See the directories GNN_NIDS_tensorflow and GNN_NIDS_IGNNITION for further details)

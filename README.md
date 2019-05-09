# DRL-SL

This is a TensorFlow implementation of the DRL-SL-based model as described in our paper:
 
Xujiang Zhao, Shu Hu, Jin-Hee Cho, Feng Chen  [Uncertainty-based Decision Making Using Deep Reinforcement Learning], FUSION (2019)

In this work, we proposed a set of uncertainty-based decision
rules to infer unknown subjective opinions by leveraging
a deep reinforcement learning (DRL) technique when a uncertain,
subjective opinion is formulated based on Subjective
Logic on graph network data. We considered three different
types of uncertainty, including vacuity, monosonance, and
dissonance, to be used as a reward in DRL with the aim of
identifying the most useful opinion paths that can lead to the
best decision making on graph network data.

![DRL-SL](example.png =250x)

![DRL-SL](drl_frame.png =250x)


## Installation

1. Clone this repository.
   ```sh
   git clone https://github.com/discovershu/DRL_SL.git
   cd DRL_SL
   ```

2. Install the dependencies. The code should run with TensorFlow 1.0 and newer.
   ```sh
   pip install -r requirements.txt  # or make install
   ```

## Requirements
* TensorFlow (1.0 or later)
* python 3.5.6
* networkx
* scikit-learn
* scipy
* numpy
* matplotlib

## Run the demo

```bash
python DRL_SL/deeppath/policy_agent.py
```

## Data

In order to use your own data, you have to provide 
* an N by N adjacency matrix (N is the number of nodes), and
* an N by D feature matrix (D is the number of features per node) -- optional

Have a look at the `get_sample_data_s()` function in `input_data.py` for an example.

In this example, we load epinioin data. The original datasets can be found here (epinion data): http://www.trustlet.org/downloaded and here (traffic data): http://inrix.com/publicsector.asp


## Model

You can choose the following model: 
* `DRL-SL-based`: policy_agent.py

## Cite

Please cite our paper if you use this code in your own work:

```
(To Appear July 2019)
```

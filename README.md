# ABIDES: Agent-Based Interactive Discrete Event Simulation environment

> ABIDES is an Agent-Based Interactive Discrete Event Simulation environment. ABIDES is designed from the ground up to support AI agent research in market applications. While simulations are certainly available within trading firms for their own internal use, there are no broadly available high-fidelity market simulation environments. We hope that the availability of such a platform will facilitate AI research in this important area. ABIDES currently enables the simulation of tens of thousands of trading agents interacting with an exchange agent to facilitate transactions. It supports configurable pairwise network latencies between each individual agent as well as the exchange. Our simulator's message-based design is modeled after NASDAQ's published equity trading protocols ITCH and OUCH. 

Please see our arXiv paper for preliminary documentation:

https://arxiv.org/abs/1904.12066

Please see the wiki for tutorials and example configurations:

https://github.com/abides-sim/abides/wiki

## Quickstart
```
mkdir project
cd project

git clone https://github.com/abides-sim/abides.git
cd abides
pip install -r requirements.txt
```


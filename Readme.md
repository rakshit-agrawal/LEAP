# LEAP 
## Learning Edge Properties in Graphs from Path Aggregations

### Abstract 
Graph edges, along with their labels, can represent information of
fundamental importance, such as links between web pages, friendship between
users, the rating given by users to other users or items, and much more. We
introduce LEAP, a trainable, general framework for predicting the presence and
properties of edges on the basis of the local structure, topology, and labels of
the graph. The LEAP framework is based on the exploration and machine-learning
aggregation of the paths connecting nodes in a graph. We provide several methods
for performing the aggregation phase by training path aggregators, and we
demonstrate the flexibility and generality of the framework by applying it to
the prediction of links and user ratings in social networks.

We validate the LEAP framework on two problems: link prediction, and user rating
prediction. On eight large datasets, among which the arXiv collaboration
network, the Yeast protein-protein interaction, and the US airlines routes
network, we show that the link prediction performance of LEAP is at least as
good as the current state of the art methods, such as SEAL and WLNM. Next, we
consider the problem of predicting user ratings on other users: this problem is
known as the edge-weight prediction problem in weighted signed networks (WSN).
On Bitcoin networks, and Wikipedia RfA, we show that LEAP performs consistently
better than the Fairness & Goodness based regression models, varying the amount
of training edges between 10 to 90%. These examples demonstrate that LEAP, in
spite of its generality, can match or best the performance of approaches that
have been especially crafted to solve very specific edge prediction problems.

### Paper

Rakshit Agrawal and Luca de Alfaro. 2019. Learning Edge Properties in Graphs from Path Aggregations. In The World Wide Web Conference (WWW '19), Ling Liu and Ryen White (Eds.). ACM, New York, NY, USA, 15-25. DOI: [https://doi.org/10.1145/3308558.3313695](https://doi.org/10.1145/3308558.3313695)

#### Bibtex
```bibtex
@inproceedings{Agrawal:2019:LEP:3308558.3313695,
 author = {Agrawal, Rakshit and de Alfaro, Luca},
 title = {Learning Edge Properties in Graphs from Path Aggregations},
 booktitle = {The World Wide Web Conference},
 series = {WWW '19},
 year = {2019},
 isbn = {978-1-4503-6674-8},
 location = {San Francisco, CA, USA},
 pages = {15--25},
 numpages = {11},
 url = {http://doi.acm.org/10.1145/3308558.3313695},
 doi = {10.1145/3308558.3313695},
 acmid = {3313695},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {Edge Learning, Neural Networks, Path Aggregation},
} 
```
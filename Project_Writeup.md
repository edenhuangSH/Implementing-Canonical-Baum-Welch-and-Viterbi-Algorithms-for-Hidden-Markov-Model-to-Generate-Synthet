
## Implementing Canonical Baum-Welch and Viterbi Algorithms for Hidden Markov Model to Generate Synthetic DNA Sequences

                                   (Dora) Dongshunyi Li, (Eden) Huang Huang

### Abstract:

Our project reviews a selected bioinformatics academic paper on Hidden Markov Model (HMM) with focus on the implementation of canonical Viterbi, forward-backward and Baum-Welch Algorithms in Python on simulated and real data. We further explore the options of codes optimization discussed in the capstone course STA 663 Statistical Computation, including the discussion of better algorithms and data structures, the JIT compilation of critical functions, vectorization and parallelism. 

Then we apply the three target algorithms on two datasets. For both datasets, we generate synthetic DNA sequences with Hidden Markov Models. The datasets were obtained by our team member Dora Li who is conducting an ongoing research in bioinformatics using the same datasets. The data source is yet to be publicized. 

The project write-up includes five sections: **Section 1 -- Background and Data** describes background of the selected research paper and the dataset for application, as well as a summary of the submitted codes and files on GitHub repository. **Section 2 -- Theory and Concepts** recaps the mathematical concepts of Hidden Markov Model the three target algorithms with examples. **Section 3 -- Implementation and Optimization** documents the improvement in performance with optimization performed. **Section 4 -- Results and Testing** shows the results on genome prediction. We also conducted a comparative analysis (speed and accuracy) with competing algorihtms from Python 3 native libary `hmmlearn` on a simulated data. Testing is illustrated in this section as well **Section 5 -- Discussion** concludes with discussion. 

# Section 1 -- Background

A Hidden Markov Model (HMM) is a statistical Markov model in which the system being modeled is assumed to be a Markov process with unobserved (hidden) states. It provides a powerful application in speech recognition, bioinformatics, knowledge discovery and clustering. 

The selected paper is *Implementing EM and Viterbi algorithms for Hidden Markov Model in Linear Memory* by Alexander Churbanov and Stephen Winters-Hilt from *BMC Bioinformatics*. This paper compares the computational expenses of canonical, checkpointing and linear implementations of Baum-Welch learning, Viterbi decoding and forward-Backword algorithms. It concludes that compared to the canonical and linear methods, the checkpointing implementations produce best overall tradeoff between memory use and speed. The linear modification replaces the array with linkedlist to store the dynamic programming backtrack pointers in viterbi algorithm and improve the efficiency of memory use. However, the original paper chooses java for implementation that has built-in linkedlist data structure. Furthermore, the linear and checkpointing methods would not boost the time performance. For simplicity purpose, our implementation in Python, which does not directly support linkedlist, focuses on the canonical method. 

The application of the three algorithms on the real data is to generate DNA mutations following a particular mutation spectra for whole-genome sequencing studies. We first train our Hidden Markov Models on sequencing data using Viterbi algorithm from cancer patients, then evaluate likelihood using forward-backword algorithm, and finally predict the models on the reference genome. The final output are the transition and emisstion matrices learned via the Baum-Welch algorithm as well as the decoding for a most possible path found by the Viterbi algorithm. 

The codes of this project is available in a public GitHub repository with a README file, an open source license, source code, test code, examples and the reproducible report. The output package `linearhmm` is installable with `python setup.py install`. `linearhmm` includes three methods:
* **utils**
    * Helper functions to generate random value given discrete distribution and encode DNA to integers
<br>
<br>
* **HMMchain**
    * Generation of simulated data through given Hidden Markov Model
<br>
<br>
* **multinomialHMM**
    * Implement Viterbi Algorithm to maximize a posteri
    * Implement Forward-Backward Algorithm for posterior marginals
    * Implement Baum Welch Algorithm for expectation maximization inference

# Section 2 -- Theory


### (i). Concepts of Hidden Markov Models

**Intuition:**
* Undirected graphical model.
* Connections between nodes indicate dependence.
* We observe $Y_1$ through $Y_n$, which we model as being observed from hidden states $S_1$ through $S_n$.
* Any particular state variable $S_k$ depends only on $S_{k−1}$ (what came before it), $S_{k+1}$ (what comes after it), and $Y_k$ (the observation associated with it).

**Model Parameter:**

* ** Transition distribution:**
    * describes the distribution for the next state given the current state.
    * $P(next State| current State)$
<br>
<br>
* ** Emission distribution:**
    * describes the distribution for the output given the current state.
    * $P(Observation|current State)$
<br>
<br>
* ** Initial state distribution:**
    * describes the starting distribution over states.
    * $P(initial State)$

### (ii). Concepts of Viterbi Algorithm

The Viterbi algorithm finds the most likely series of states, given some observations and assumed parameters.

**Intuition:**
* Efficient way of finding the most likely state sequence.
* Method is general statistical framework of compound decision theory. 
* Maximizes a posteriori probability recursively.
* Assumed to have a finite-state discrete-time Markov process.

**Pseudocodes and Equations:**

* Maximum a posteri (MAP) probability, given by: <br>
$$P(States|Observations) = \dfrac{P(Observations)| States)P(States)}{P(Observations)}$$

* Given a hidden Markov model (HMM) with:
  * State space S
  * Initial probabilities $\pi_i$ of being in state $i$
  * Transition probabilities $a_{i,j}$ of transitioning from state i to state j.
  * We observe outputs $y_1$,$\dots$, $y_T$.
  * The most likely state sequence $x_1$,$\dots$,$x_T$ that produces the observations is given by the recurrence relations:
  $$ V_{1,k} = P(y_1 | k) \cdot \pi_k$$

$$ V_{t,k} = max_x(P(y_t|k)\cdot a_{x,k} \cdot V_{t-1,x}$$

* $V_{t,k}$ is the probability of the most probable state sequence $\mathrm{P}\big(x_1,\dots,x_T,y_1,\dots, y_T\big)$
* The Viterbi path can be retrieved by saving back pointers that remember which state x was used in the second equation.
* Let $\mathrm{Ptr}(k,t)$ be the function that returns the value of x used to compute $V_{t,k}$ if $t > 1$, or $k$ if $t=1$. Then:

$$ x_T = argmax_x(V_{T,x})$$

$$ x_{t-1} = Ptr(x_{t}, t)$$


### (iii). Concepts of Forward-backward Algorithm

The goal of the forward-backward algorithm is to find the conditional distribution over hidden states given the data.

**Intuition:**
* It is used to find the most likely state for any point in time.
* It cannot, however, be used to find the most likely sequence of states 

**Pseudocodes and Intuition:**
* Computes posterior marginals of all hidden state variables given a sequence of observations/emissions.
* Computes, for all hidden state variables $S_k \in \{S_1, \dots, S_t\}$, the distribution $P(S_k\ |\ o_{1:t})$. This inference task is usually called smoothing.
* The algorithm makes use of the principle of dynamic programming to compute efficiently the values that are required to obtain the posterior marginal distributions in two passes.
* The first pass goes forward in time while the second goes backward in time.

### (iv). Concepts of Baum Welch Algorithm:

Baum–Welch algorithm is used to infer unknown parameters of a Hidden Markov Model. It makes use of the forward-backward algorithm to update the hypothesis.

**Model Parameters:**
* Initial State Probabilities
* Transition Matrix
* Emission Matrix

**Pseudocodes and Equations:**

* Calculate the temporary variables, according to Bayes' theorem: <br>
Temporary variables are the probabilities of being in state $i$ at time $t$ given the observed sequence $Y$ and the parameters $\theta$
$$\gamma_i(t)=P(X_t=i|Y,\theta) = \frac{\alpha_i(t)\beta_i(t)}{\sum_{j=1}^N \alpha_j(t)\beta_j(t)}$$
The probability of being in state $i$ and $j$ at times $t$ and $t+1$ respectively given the observed sequence $Y$ and parameters $\theta$:
$$\xi_{ij}(t)=P(X_t=i,X_{t+1}=j|Y,\theta)=\frac{\alpha_i(t) a_{ij} \beta_j(t+1) b_j(y_{t+1})}{\sum_{k=1}^N \alpha_k(T)}$$

* Update Inital State Probability: <br>
Initial state probabilities are the expected frequency spent in state i at time 1:
$$\pi_i^* = \gamma_i(1)$$

* Update Transition Matrix: <br>
Transition matrix describes the expected number of transitions from state $i$ to state $j$ compared to the expected total number of transitions away from state $i$:
$$ a_{ij}^*=\frac{\sum^{T-1}_{t=1}\xi_{ij}(t)}{\sum^{T-1}_{t=1}\gamma_i(t)}$$

* Update Emission Matrix: <br>
Emission matrix documents the expected number of times the output observations have been equal to $v_k$ while in state $i$ over the expected total number of times in state $i$:
$$b_i^*(v_k)=\frac{\sum^T_{t=1} 1_{y_t=v_k} \gamma_i(t)}{\sum^T_{t=1} \gamma_i(t)}$$



### (v). Example: DNA Data

DNA is composed of different states of nucleotides, i.e. A, C, T, G. Publications have shown that contiguous nucleotides are highly dependent with each other. In other words, a nucleotide determines if the nucleotide after it is A, C, T or G. Therefore, the generation of synthetic DNA can be modeled as a Markov chain, where there are 4 possible observations. However, it was further revealed that the emission probabilities in different regions of DNA is different. For example, the probability for a A after a C is different in A-G rich region and in C-T rich region. In this case, a Hidden Markov Model is more appropriate where there are 2 states $S = {S_1, S_2}$, with $S_1$ and $S_2$ representing A-G rich and C-T rich. Each state has 4 possible observations and the emission probabilities can be modeled as $b_j(o_t) = p(o_t|q_t=S_j)$. Here $1 \leq j \leq N$, where N is the total number of states and equal to 2 in our case. $t$ is a time point and we have $1 \leq t \leq T$, where T is the total number of time points. $q_t$ is a realized state being visited at time $t$. 

The operations mainly involve manipulations of DNA strings. Operations on strings (e.g. comparison, concatenation) are more costly in terms of both time and space, compared to algorithmtic operations. Therefore, one approach is to encode DNA sequences as binaries or integers. A single nucleotide can be encoded as a 2-digit binary string or a single integer. In this way, all string manipulations can be converted to algorithmic operations. The relationship between nucleotide and binary/decimal representation is shown below: 

| Nucleotides   | Binary String | Integer|
| ------------- |:-------------:| -----:|
| T | 00 | 0 |
| C | 01 | 1 |
| A | 10 | 2 |
| G | 11 | 3 |

Following the modeling methods described above, we applied our implementations of the Baum-Welch algorithm and the Viterbi algorithm on a simulated data set and on a real data set. We will show the transition and emisstion matrixes learned via the Baum-Welch algorithm and also the decoding for a most possible path found by the Viterbi algorithm.

# Section 3 -- Implementation and Optimization










```python

```


```python

```

# Section 4 -- Results and Testing











## ( ). Summary of Results







```python

```

## ( ). Comparative Analysis with `hmmlearn`

Let's consider the following simple HMM.
* Composed of 2 hidden states: Healthy and Fever.
* Composed of 3 possible observation: Normal, Cold, Dizzy

The model can then be used to predict if a person is feverish at every timestep from a given observation sequence. There are several paths through the hidden states (Healthy and Fever) that lead to the given sequence, but they do not have the same probability.

<img src="http://iacs-courses.seas.harvard.edu/courses/am207/blog/hmm1.png" />

(**source**: Havard AM207 Course Website)

#### -- Implementation on `linearhmm` (our algorithms):

##### a) Generate simulated data set


```python
import random
import numpy as np
import linearhmm
np.random.seed(1)

states = ('Healthy', 'Fever')
observations = ('cold', 'normal', 'normal')
start_probability = {'Healthy': 0.6, 'Fever': 0.4}

transition_probability = {
   'Healthy' : {'Healthy': 0.69, 'Fever': 0.3, 'E': 0.01},
   'Fever' : {'Healthy': 0.4, 'Fever': 0.59, 'E': 0.01},
   }
 
emission_probability = {
   'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
   'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
   }

N = 100 # 100 samples
hidden, visible = linearhmm.simulate_data(N, states, observations, start_probability, transition_probability, emission_probability )
visible
```

##### b) Get the accuracy from Viterbi Algorithm:


```python
(prob, p_hidden) = linearhmm.viterbi(visible,
                   states,
                   start_probability,
                   transition_probability,
                   emission_probability)

# assess accuracy of the HMM model:
wrong= 0
for i in range(len(hidden)):
    if hidden[i] != p_hidden[i]:
        wrong = wrong + 1
print ("accuracy: " + str(1-float(wrong)/N))
```

##### c) Get the transition and emission matrixes learned by Baum-Welch algorithm:


```python
A_mat, O_mat = linearhmm.baum_welch(num_states=2,num_obs=4,observ=np.array(visible))
print(A_mat)
print(O_mat)
```


    

    NameErrorTraceback (most recent call last)

    <ipython-input-2-e1e1017e094a> in <module>()
    ----> 1 A_mat, O_mat = linearhmm.baum_welch(num_states=2,num_obs=4,observ=np.array(visible))
          2 print(A_mat)
          3 print(O_mat)


    NameError: name 'linearhmm' is not defined


#### -- Implementation on `hmmlearn` (Python 3 library):

##### a) Generate simulated data set


```python
import random
import numpy as np
import hmmlearn as hmm
np.random.seed(1)

# same model parameters as previous:
model = hmm.MultinomialHMM(n_components=2) # 2 states
model.startprob_ = np.array([0.6, 0.4])
model.transmat_ = np.array([[0.69, 0.3, 0.01],
                            [0.4, 0.59, 0.01]])
model.emissionprob_ = np.array([[0.5, 0.4, 0.1],
                                [0.1, 0.3, 0.6]])

# Predict the optimal sequence of internal hidden state
X = np.atleast_2d([3, 4, 5, 6, 7]).T
print(model.decode(X))
```


    

    AttributeErrorTraceback (most recent call last)

    <ipython-input-5-5cfdfff1101a> in <module>()
          5 
          6 # same model parameters as previous:
    ----> 7 model = hmm.MultinomialHMM(n_components=2) # 2 states
          8 model.startprob_ = np.array([0.6, 0.4])
          9 model.transmat_ = np.array([[0.69, 0.3, 0.01],


    AttributeError: module 'hmmlearn' has no attribute 'MultinomialHMM'


##### b) Get the accuracy from Viterbi Algorithm:


```python
!pip update hmmlearn
```

    ERROR: unknown command "update"



```python

```

##### c) Get the transition and emission matrixes learned by Baum-Welch algorithm:


```python

```


```python

```

## ( ). Codes Testing


```python

```


```python

```


```python

```


```python

```

# Section 5 -- Discussion











Our two team members, Dora Li and Eden Huang split all the tasks evenly throughout the project. Both team members worked together on algorithms coding, literature review and junit testing. Dora was also responsible for codes optimization, parallel programming and real data application, while Eden focused on concepts overview, comparative analysis project write-up and results reporting. 


```python

```

# Reference:

1.Churbanov, A., & Winters-Hilt, S. (2008). "Implementing EM and Viterbi algorithms for Hidden Markov Model in linear memory". BMC Bioinformatics, 9(224). Retrieved April 20, 2017.

2.Yuan, J.J. (2014, Jan 22). "VITERBI ALGORITHM: FINDING MOST LIKELY SEQUENCE IN HMM" [Web log post]. Retrieved from https://jyyuan.wordpress.com/2014/01/28/baum-welch-algorithm-finding-parameters-for-our-hmm/

3.Yuan, J.J. (2014, Jan 26). "FORWARD-BACKWARD ALGORITHM: FINDING PROBABILITY OF STATES AT EACH TIME STEP IN AN HMM" [Web log post]. Retrieved from
https://jyyuan.wordpress.com/2014/01/26/forward-backward-algorithm-finding-probability-of-states-at-each-time-step-in-an-hmm/

4.Yuan, J.J. (2014, Jan 28). "BAUM-WELCH ALGORITHM: FINDING PARAMETERS FOR OUR HMM" [Web log post]. Retrieved from
https://jyyuan.wordpress.com/2014/01/28/baum-welch-algorithm-finding-parameters-for-our-hmm/

5.Baum, L. E.; Petrie, T. (1966). "Statistical Inference for Probabilistic Functions of Finite State Markov Chains". The Annals of Mathematical Statistics. 37 (6): 1554–1563. doi:10.1214/aoms/1177699147. Retrieved 28 April 2017.

6.Rabiner, Lawrence. "First Hand: The Hidden Markov Model". IEEE Global History Network. Retrieved 28 April 2017.

7.Protopapas, Pavlos. Lecture 18 and Lecture 19, AM207 Spring 2014. Havard Extension course website. Retrieved from http://iacs-courses.seas.harvard.edu/courses/am207/blog/lecture-18.html


```python

```


```python

```


```python

```


```python

```






### Background

State the research paper you are using. Describe the concept of the algorithm and why it is interesting and/or useful. If appropriate, describe the mathematical basis of the algorithm. Some potential topics for the backgorund include:

- What problem does it address? 
- What are known and possible applications of the algorithm? 
- What are its advantages and disadvantages relative to other algorithms?
- How will you use it in your research?

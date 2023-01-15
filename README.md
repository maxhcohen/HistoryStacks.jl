# Overview
This is a lightweight package containing utilities for online parameter estimation of linear regression models of the form $$ y(t) = F(t)\theta, $$ where $\theta$ is a vector of uncertain parameters, $t$ represents time, $y$ is a target vector, and $F$ is a regression matrix. The main type exported by this package is a `HistoryStack` denoted as $$ H = \{Y_j, F_j\}_{j=1}^M $$ containing pairs of targets and regressors recorded at various time instances. The main utility of this package is an implementation of the algorithm originally proposed in

* G. Chowdhary and E. Johnson, "A singular value maximizing data recording algorithm for concurrent learning," in Proceedings of the American Control Conference, pp. 3547-3552, 2011,

for managing data in the history stack, which can then be used in applications such as adaptive control. For further details, see the examples folder of this repo.

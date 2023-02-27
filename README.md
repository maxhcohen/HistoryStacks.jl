# Overview
This is a lightweight package containing utilities for online parameter estimation of linear regression models. The main type exported by this package is a `HistoryStack` containing pairs of targets and regressors recorded at various time instances. The main utility of this package is an implementation of the algorithm originally proposed in

* G. Chowdhary and E. Johnson, "A singular value maximizing data recording algorithm for concurrent learning," in Proceedings of the American Control Conference, pp. 3547-3552, 2011,

for managing data in the history stack, which can then be used in applications such as adaptive control. For more details on these utilities please check out the examples folder.

module HistoryStacks

# Import required modules
using LinearAlgebra

# Export stuff
export HistoryStack
export stacksum
export update! 
export worst_case_error
export gradient_vector_field
export upper_bound_update
export smid_update

# Source code
include("history_stack.jl")

end # module HistoryStacks

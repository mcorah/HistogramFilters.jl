module HistogramFilters

export AbstractHistogramFilter

export HistogramFilter, duplicate, reset_distribution, ndim, get_data, get_data,
  get_buffer, swap_buffer!, copy_filter!, get_range, generate_prior,
  test_histogram, weighted_average, size

export SparseHistogramFilter, to_sparse, set_threshold!, drop_below_threshold!,
  sparsity

abstract type AbstractHistogramFilter end

# Returns the data values themselves (as in for sparse matrices)
get_data(x::AbstractHistogramFilter) = x.data
get_buffer(x::AbstractHistogramFilter) = x.buffer

get_range(x::AbstractHistogramFilter) = x.range
get_range(x::AbstractHistogramFilter, index) = x.range[index]

function swap_buffer!(x::AbstractHistogramFilter)
  old_data = x.data

  x.data = x.buffer
  x.buffer = old_data
end

ndim(x::AbstractHistogramFilter) = length(x.range)

include("dense_histograms.jl")
include("sparse_histograms.jl")

end # module

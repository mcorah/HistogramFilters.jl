using LinearAlgebra

import Base.size

mutable struct HistogramFilter{RangeType <: Real} <: AbstractHistogramFilter
  range::Tuple{Vector{RangeType},Vector{RangeType}}
  data::Array{Float64, 2}

  buffer::Array{Float64, 2}

  # The default constructor specializes for when we can infer the type of the
  # range from the signature. (Alternatively, see the outer constructor).
  function HistogramFilter(range::Tuple{Vector{D}, Vector{D}}, data) where D <: Real

    new{D}(range, data, Array{Float64}(undef, size(data)))
  end
end

# Outer constructor that defers determination of the histogram type until after
# pulling the ranges
HistogramFilter(range, data) = HistogramFilter(map(collect, range), data)

HistogramFilter(range) = HistogramFilter(range, generate_prior(range))

function reset_distribution!(x::HistogramFilter)
  x.data = generate_prior(x)
  Void
end

# Copy constructor. Note that this only duplicates the data.
# We assume that nobody is crazy enough to modify the range.
duplicate(x::HistogramFilter) = HistogramFilter(x)
function HistogramFilter(x::HistogramFilter)
  HistogramFilter(get_range(x), Array(get_data(x)))
end

size(x::HistogramFilter) = size(x.data)

# Returns the data matrix
get_data(x::HistogramFilter) = x.data

# Copies filter data (exclusively)
function copy_filter!(x::HistogramFilter; out::HistogramFilter)
  out.data .= x.data
end

function generate_prior(range)
  lengths = map(length, range)
  num_cells = prod(lengths)

  ones(lengths) / num_cells
end
generate_prior(x::HistogramFilter) = generate_prior(x.range)

weighted_average(x::HistogramFilter, dim) = dot(sum_all_dims_but(x.data, dim), collect(x.range[dim]))
weighted_average(h::HistogramFilter) = map(x->weighted_average(h, x), 1:length(h.range))

# Maps indices in the ranges to the space they represent
function from_indices(h::HistogramFilter, inds)
  n = length(h.range)
  out = Array{Float64}(undef, n)
  for ii = 1:n
    out[ii] = h.range[ii][inds[ii]]
  end
  out
end

###########
# Test code
###########

test_total_probability(x::HistogramFilter) = abs(sum(get_data(x)) - 1) < 1e-3

test_positive(x::HistogramFilter) = all(get_data(x) .>= 0)

function test_histogram()
  h = HistogramFilter((1:0.5:2, 1:0.5:2))

  assert(test_total_probability(h))
  assert(test_positive(h))
end

function sum_all_dims_but(data::Array{T,N}, dim) where {T, N}
  all_but = filter(x->x!=dim, 1:N)
  sum(data; dims=all_but)[:]
end

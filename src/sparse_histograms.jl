using SparseArrays

# Construct a sparse vector based on the input data
function to_sparse(x; threshold)
  ret = spzeros(length(x))
  for (ii, val) in enumerate(x)
    if val > threshold
      push!(ret.nzind, ii)
      push!(ret.nzval, val)
    end
  end

  ret
end

mutable struct SparseHistogramFilter{RangeType <: Real} <: AbstractHistogramFilter
  range::Tuple{Vector{RangeType},Vector{RangeType}}
  data::SparseVector{Float64, Int64}

  buffer::SparseVector{Float64, Int64}

  # Threshold defines the minimum value to keep in the filter
  threshold::Float64

  # The default constructor specializes for when we can infer the type of the
  # range from the signature. (Alternatively, see the outer constructor).
  function SparseHistogramFilter(range::Tuple{Vector{D}, Vector{D}},
                           data::SparseVector;
                           threshold) where D <: Real
    new{D}(range, data, spzeros(length(data)), threshold)
  end
end

# Constructor for when the input data is not sparse
function SparseHistogramFilter(range, data; threshold) <: AbstractHistogramFilter
  SparseHistogramFilter(range, to_sparse(data, threshold=threshold);
                  threshold=threshold)
end

# Outer constructor that defers determination of the histogram type until after
# pulling the ranges
function SparseHistogramFilter(range, data::SparseVector; threshold)
  SparseHistogramFilter(map(collect, range), data; threshold=threshold)
end

# Copy constructor. Note that this only duplicates the data.
# We assume that nobody is crazy enough to modify the range.
duplicate(x::SparseHistogramFilter) = SparseHistogramFilter(x)
# Note, this will fail if no threshold is given when converting a
# HistogramFilter
function SparseHistogramFilter(x::AbstractHistogramFilter; threshold = x.threshold)
  SparseHistogramFilter(get_range(x), to_sparse(get_data(x), threshold=threshold),
                  threshold=threshold)
end

set_threshold!(x::SparseHistogramFilter; threshold) = (x.threshold = threshold)

# Returns the data matrix
size(x::SparseHistogramFilter) = map(length, x.range)
get_data(x::SparseHistogramFilter) = reshape(x.data, size(x))

#
# Sparse helper methods
#

function clear!(x::SparseVector)
  resize!(x.nzind, 0)
  resize!(x.nzval, 0)
end

function copy_sparse!(out, in, threshold)
  nzind = in.nzind
  nzval = in.nzval
  n = length(nzind)

  out_ind = out.nzind
  out_val = out.nzval

  # Presume that we will copy most of the input
  resize!(out_ind, n)
  resize!(out_val, n)

  # Copy values above the threshold
  end_ind = 1
  @inbounds for ii in 1:n
    if nzval[ii] > threshold
      out_ind[end_ind] = nzind[ii]
      out_val[end_ind] = nzval[ii]

      end_ind += 1
    end
  end

  resize!(out_ind, end_ind-1)
  resize!(out_val, end_ind-1)

  nothing
end

# Copies filter data, keeping nonzeros
function copy_filter!(x::SparseHistogramFilter; out::SparseHistogramFilter)
  copy_sparse!(out.data, x.data, x.threshold)
  clear!(out.buffer)

  out.threshold = x.threshold
end

# Remove values from the histogram below a given threshold
#
# Trim resizes the matrix. We will generally continue to operate in place so we
# will end up using the empty space
function drop_below_threshold!(x::SparseHistogramFilter;
                               threshold = x.threshold,
                               trim = true
                              )
  droptol!(x.data, threshold, trim = trim)
  droptol!(x.buffer, threshold, trim = trim)
end
drop_below_threshold!(x::HistogramFilter; kwargs...) = nothing

sparsity(x::SparseHistogramFilter) =  1.0 - nnz(get_data(x)) / length(get_data(x))

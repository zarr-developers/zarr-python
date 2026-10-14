**Breaking:** a v3 array's sharding codec is judged against the shard it
is handed, and its two pipelines are read as the array's is. Its
`chunk_shape` has an inner chunk length for each axis of the shard,
dividing each length the shards take along it -- the shard being the
chunk the codec is handed, so a transposed shard is divided along its
transposed axes. Its inner codecs are handed the inner chunks, of the
shard's data type, and its index codecs the shard index, of `uint64`,
with an axis more than the shard. A shard that does not divide, or a
pipeline that does not fit what it is handed -- an index `bytes` codec
without an `endian` -- has a problem where it sits, where the package
accepted it before. A codec definition
says what the pipelines it holds are handed, `pipelines`, and each
codec's `Stage` keeps the stages of those pipelines as `inner`.

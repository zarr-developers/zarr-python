The types of every extension (each codec's, data type's, chunk grid's and
chunk key encoding's name constants, `Literal` types, configuration and object
`TypedDict`s and metadata aliases, in both formats) now live in private modules
that hold types only and import no definition. A definition of one kind can
therefore import the types of another kind at run time, as a data type that
names its default codec does. Every name keeps its public import paths, since
each extension's module re-exports its types.

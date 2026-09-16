# ndsel conformance corpus

Language-agnostic fixtures. Each file is a JSON array of cases.

A **success** case:
    { "name": "...", "input": <message>, "normalized": <canonical transform without `kind`> }

An **error** case:
    { "name": "...", "input": <message>, "error": "<reason_code>" }

An implementation is conformant iff, for every success case,
`normalize(input)` equals `normalized` by structural JSON equality, and for
every error case, `normalize(input)` is rejected with the given reason code.

The `normalized` value is a canonical `transform` body (the `kind` field is
omitted; implementations compare the transform structure).

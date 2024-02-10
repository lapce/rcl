# Types

***Warning**: The type system is a work in progress. In particular, record types
are missing, and the implementation of the typechecker is too ad-hoc.*

RCL has a type system that can help to prevent bugs and make configuration more
self-documenting.

 * **The type system is gradual.** You can add type annotations to variables,
   and those annotations will be enforced.[^1] Variables that are not annotated
   are unconstrained.
 * **The type system is static.** Type errors in unreachable code will be
   reported. Sufficient type information needs to be available at typecheck time
   to be able to prove that an expression contains a type error. When not all
   types are known statically, typechecks are deferred to runtime.
 * **The type system is structural, not nominal.** Two different type aliases of
   the same type can be used interchangeably.

[^1]: Unlike in Python, where external tools can statically typecheck type
      annotations, but they are ignored by the runtime.

Type annotations can be applied at [let bindings](syntax.md#let-bindings) by
adding a colon after the name:

```rcl
let x: Int = 42; x
```

The names of all types start with a capital letter.

## Primitive types

The primitive types are:

 * `Bool`, the type of `true` and `false`.
 * `Null`, the type of `null`.
 * `String`, the type of strings.
 * `Int`, the signed integer type.
 * Currently there is no type for non-integer numbers. (TODO: Add one.)

## Collection types

There are three built-in generic collection types:

 * `Dict[K, V]`, a dictionary with keys of type `K` and values of type `V`.
 * `List[T]`, a list with elements of type `T`.
 * `Set[T]`, a set with elements of type `T`.

For example, we can annotate this dict as follows:

```rcl
let port_names: Dict[Int, String] = {
  22: "ssh",
  80: "http",
  443: "https",
};
```

## Function types

Function types are written as an argument list between parentheses, a thin
arrow, and then the result type:

```rcl
let add: (Int, Int) -> Int = (x, y) => x + y;
```

The parentheses are mandatory, even for functions that take a single argument.
A trailing comma is optional.

## The dynamic type

The `Dynamic` type signals to the typechecker that the type of a value is not
known statically. Variables that have type `Dynamic` never cause static type
errors, but they can still cause runtime type errors.

```rcl
let x: Int = 32;
let y: Dynamic = x;
// Not a static type error: an expression with type Dynamic could evaluate to
// a string, so assigning it to a variable of type String is allowed. But at
// runtime, we verify that the value is really a string, and that check fails.
let z: String = y;
z
```

Annotating a variable with `Dynamic` is not useful. At best it behaves the same
as not annotating the variable, but in the worst case it forces the typechecker
to discard type information that it was able to infer. However, `Dynamic` can be
useful as part of a more complex type, to partially enforce some structure:

```rcl
let widgets: Dict[String, Dynamic] = {
  frobnicator = { foobar = 42 },
  turbo-encabulator = { prefabulated = true, bearings = "spurving" },
};
```

TODO: Currently the `Dynamic` type cannot be written directly.
// RCL -- A reasonable configuration language.
// Copyright 2023 Ruud van Asseldonk

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// A copy of the License has been included in the root of the repository.

//! Type requirements, a central building block of typechecking.

use std::rc::Rc;

use crate::error::{IntoError, PathElement, Result};
use crate::runtime::Value;
use crate::source::Span;
use crate::types::Type;

/// The result of a static typecheck.
///
/// A diff can represent type errors, nested type errors, no error, or a signal
/// that the check could not be performed statically and needs to be deferred to
/// runtime.
pub enum TypeDiff {
    /// No error. The actual type matches the expected type.
    Ok(Type),

    /// The check could not be performed statically, a runtime check is needed.
    Defer,

    /// A static type mismatch that cannot be broken down further.
    Error {
        /// The type that was expected here.
        expected: ReqType,
        /// The reason the type was expected.
        reason: Reason,
        /// The type that we encountered instead.
        actual: Type,
    },

    /// There is a type mismatch in the element type of a list.
    List(Box<TypeDiff>),

    /// There is a type mismatch in the element type of a set.
    Set(Box<TypeDiff>),

    /// There is a type mismatch somewhere in the dict type.
    Dict {
        key: Box<TypeDiff>,
        value: Box<TypeDiff>,
    },
    // TODO: Define function type errors,
}

/// Context for a type requirement ([`TypeReq`]).
///
/// The context explains why the particular requirement is there. In a type
/// error, it controls how the error manifests.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Reason {
    /// The type was required due to a type annotation.
    Annotation(Span),

    /// A boolean was required because it's used as a condition.
    ///
    /// The span points at the syntactic element that expects a condition (an
    /// `if` or an `assert`).
    Condition(Span),

    /// The type was required due to an operator.
    Operator(Span),
}

/// A type requirement.
///
/// A [`Type`] is a type that the typechecker inferred. A [`TypeReq`] is a
/// requirement that the typechecker needs to fulfill. Type requirements
/// correspond to types, but carry additional context about _why_ that type was
/// expected in a particular location. (Context such as “because it was part of
/// a type annotation at this span”, or “because conditions should be booleans”.)
/// Requirements can be fulfilled by subtypes of the required type.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum TypeReq {
    /// No constraints are made on the type, any value will suffice.
    None,

    /// The type has to be a subtype of the required type for the given reason.
    Type(ReqType, Reason),
}

/// The types that can occur in type requirements.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum ReqType {
    /// Require [`Type::Bool`].
    Bool,

    /// Require [`Type::Int`].
    Int,

    /// Require [`Type::Null`].
    Null,

    /// Require [`Type::String`].
    String,

    /// Require [`Type::List`].
    List(Rc<TypeReq>),

    /// Require [`Type::Set`].
    Set(Rc<TypeReq>),

    /// Require [`Type::Dict`].
    Dict(Rc<DictReq>),

    /// Require [`Type::Function`].
    Function(Rc<FunctionReq>),
}

/// The type parameter requirements for the `Dict` type.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct DictReq {
    pub key: TypeReq,
    pub value: TypeReq,
}

/// A function type requirement.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct FunctionReq {
    pub args: Vec<TypeReq>,
    pub result: TypeReq,
}

impl TypeReq {
    /// Statically check that the given type is a subtype of the required type.
    pub fn check_type(&self, type_: &Type) -> TypeDiff {
        let (req_type, reason) = match self {
            // If there was no requirement, any type is acceptable.
            TypeReq::None => return TypeDiff::Ok(type_.clone()),
            TypeReq::Type(req_type, reason) => (req_type, reason),
        };
        match (req_type, type_) {
            // If there was some requirement, but we don't know the type,
            // then we have to defer the typecheck to runtime.
            (_, Type::Dynamic) => TypeDiff::Defer,

            // For the primitive types, we just check for matching values.
            (ReqType::Null, Type::Null) => TypeDiff::Ok(type_.clone()),
            (ReqType::Bool, Type::Bool) => TypeDiff::Ok(type_.clone()),
            (ReqType::Int, Type::Int) => TypeDiff::Ok(type_.clone()),
            (ReqType::String, Type::String) => TypeDiff::Ok(type_.clone()),

            // For compound types, we need to do the inner inspection.
            (ReqType::List(elem_req), Type::List(elem_type)) => {
                match elem_req.check_type(elem_type) {
                    TypeDiff::Ok(..) => TypeDiff::Ok(type_.clone()),
                    TypeDiff::Defer => TypeDiff::Defer,
                    error => TypeDiff::List(error.into()),
                }
            }
            (ReqType::Set(elem_req), Type::Set(elem_type)) => {
                match elem_req.check_type(elem_type) {
                    TypeDiff::Ok(..) => TypeDiff::Ok(type_.clone()),
                    TypeDiff::Defer => TypeDiff::Defer,
                    error => TypeDiff::Set(error.into()),
                }
            }
            (ReqType::Dict(kv_req), Type::Dict(kv_type)) => {
                let k_diff = kv_req.key.check_type(&kv_type.key);
                let v_diff = kv_req.value.check_type(&kv_type.value);
                match (k_diff, v_diff) {
                    (TypeDiff::Ok(..), TypeDiff::Ok(..)) => TypeDiff::Ok(type_.clone()),
                    (TypeDiff::Defer, _) => TypeDiff::Defer,
                    (_, TypeDiff::Defer) => TypeDiff::Defer,
                    (k_diff, v_diff) => TypeDiff::Dict {
                        key: k_diff.into(),
                        value: v_diff.into(),
                    },
                }
            }

            // If we did not match anything, then this is a type error.
            _ => TypeDiff::Error {
                expected: req_type.clone(),
                reason: reason.clone(),
                actual: type_.clone(),
            },
        }
    }

    /// Dynamically check that the given value fits the required type.
    pub fn check_value(&self, at: Span, value: &Value) -> Result<()> {
        let (req_type, reason) = match self {
            TypeReq::None => unreachable!("TypeReq::None should not be checked at runtime."),
            TypeReq::Type(req_type, reason) => (req_type, reason),
        };
        match (req_type, value) {
            // For the primitive types, we just check for matching values.
            (ReqType::Null, Value::Null) => Ok(()),
            (ReqType::Bool, Value::Bool(..)) => Ok(()),
            (ReqType::Int, Value::Int(..)) => Ok(()),
            (ReqType::String, Value::String(..)) => Ok(()),

            // For compound types, we descend into them to check.
            (ReqType::List(elem_type), Value::List(elems)) => {
                for (i, elem) in elems.iter().enumerate() {
                    elem_type
                        .check_value(at, elem)
                        .map_err(|err| err.with_path_element(PathElement::Index(i)))?;
                }
                Ok(())
            }
            (ReqType::Set(elem_type), Value::Set(elems)) => {
                for (i, elem) in elems.iter().enumerate() {
                    elem_type.check_value(at, elem).map_err(|err|
                        // Even though sets don't strictly have indexes,
                        // they do have an order, so report the index to
                        // clarify that this is a nested error.
                        err.with_path_element(PathElement::Index(i)))?;
                }
                Ok(())
            }
            (ReqType::Dict(dict), Value::Dict(kvs)) => {
                for (k, v) in kvs.iter() {
                    dict.key.check_value(at, k).map_err(|err| {
                        err.with_path_element(PathElement::Key("TODO: Support any key".into()))
                    })?;
                    dict.value.check_value(at, v).map_err(|err| {
                        err.with_path_element(PathElement::Key("TODO: Support any key".into()))
                    })?;
                }
                Ok(())
            }

            // TODO: Typecheck functions.
            _ => {
                let _ = reason;
                at.error("Type mismatch in value. TODO: Pretty-print.")
                    .err()
            }
        }
    }
}

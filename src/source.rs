// RCL -- A reasonable configuration language.
// Copyright 2023 Ruud van Asseldonk

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// A copy of the License has been included in the root of the repository.

//! Types for dealing with input source code.

use std::fmt;

/// A named input document.
pub struct Doc<'a> {
    /// Path can be a file path, but also a name such as "stdin".
    pub name: &'a str,

    /// The contents of the file.
    pub data: &'a str,

    /// the line offset of the file
    pub line_offset: usize,
}

/// A list of input documents.
pub type Inputs<'a> = [Doc<'a>];

/// The index of a document in the list of input files.
#[derive(Copy, Clone, Eq, Ord, PartialEq, PartialOrd)]
pub struct DocId(pub u32);

// coverage:off -- Only used in assertion failures, which should be uncovered.
impl fmt::Debug for DocId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
// coverage:on

/// Marks a location in a source file by byte offset.
///
/// Invariant: the start and end offset of the span should be a char boundary in
/// the source document. That is, a span should never slice a code point in half.
/// The lexer contains debug assertions to verify this.
#[derive(Copy, Clone, Eq, Ord, PartialEq, PartialOrd)]
pub struct Span {
    /// Packed fields, use constructor and getter methods to pack/unpack.
    ///
    /// From least significant to most significant bit:
    /// * 48 bits start offset (6 bytes)
    /// * 48 bits end offset (6 bytes)
    /// * 32 bits document id (4 bytes)
    ///
    /// While 32 bit offset may be too constraining for a language that can also
    /// be a data format, 48 bits of offset enables documents of 252 TiB, which
    /// ought to be enough for a single document.
    data: u128,
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "d{}[{}..{}]", self.doc().0, self.start(), self.end())
    }
}

impl Span {
    #[inline(always)]
    pub fn new(doc: DocId, start: usize, end: usize) -> Span {
        // We could turn this into a proper error and report it, but it would
        // make things really tedious. It's one of those things where if you
        // don't fix it a fuzzer will force you to, except in this case we need
        // an input larger than 262 TiB before it happens, so we can get away
        // with a panic.
        assert!(
            end as u128 <= 0xffff_ffff_ffff,
            // coverage:off -- Error is not tested, I don't have such large disk.
            "Document should not be larger than 262 TiB.",
            // coverage:on
        );
        debug_assert!(end >= start);
        Span {
            data: (start as u128) | ((end as u128) << 48) | ((doc.0 as u128) << 96),
        }
    }

    /// Id of the document that this span belongs to.
    #[inline(always)]
    pub fn doc(&self) -> DocId {
        DocId(((self.data >> 96) & 0xffff_ffff) as u32)
    }

    /// Start byte offset of the span, inclusive.
    #[inline(always)]
    pub fn start(&self) -> usize {
        (self.data & 0xffff_ffff_ffff) as usize
    }

    /// End byte offset of the span, exclusive.
    #[inline(always)]
    pub fn end(&self) -> usize {
        ((self.data >> 48) & 0xffff_ffff_ffff) as usize
    }

    /// Length of this span in bytes.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.end() - self.start()
    }

    /// Return the slice from the input that the span spans.
    pub fn resolve<'a>(&self, input: impl Source<'a>) -> &'a str {
        input.resolve(*self)
    }

    pub fn start_line(&self, input: &str) -> usize {
        // Locate the line that contains the error.
        let mut line = 1;
        let mut line_start = 0;
        let start = self.start();
        for (&c, i) in input.as_bytes().iter().zip(0..) {
            if i == start {
                break;
            }
            if c == b'\n' {
                line += 1;
                line_start = i + 1;
            }
        }
        for (&c, _) in input.as_bytes()[line_start..].iter().zip(line_start..) {
            if c == b'\n' {
                break;
            }
        }
        line
    }

    /// Delete n bytes from the start of the span.
    pub fn trim_start(&self, n: usize) -> Span {
        let n_trim = self.len().min(n);
        Span::new(self.doc(), self.start() + n_trim, self.end())
    }

    /// Return a span that runs from self up to but not including `other`.
    pub fn until(&self, other: Span) -> Span {
        debug_assert_eq!(self.doc(), other.doc());
        debug_assert!(other.start() >= self.end());
        Span::new(self.doc(), self.start(), other.start())
    }

    /// Return a span that encloses both spans.
    pub fn union(&self, other: Span) -> Span {
        debug_assert_eq!(self.doc(), other.doc());
        Span::new(
            self.doc(),
            self.start().min(other.start()),
            self.end().max(other.end()),
        )
    }
}

pub trait Source<'a> {
    /// Return the slice from the input that the span spans.
    fn resolve(self, span: Span) -> &'a str;
}

/// If we resolve against a string, then we assume that this string is the
/// right document for this span, and we ignore the document id in the span.
impl<'a> Source<'a> for &'a str {
    fn resolve(self, span: Span) -> &'a str {
        &self[span.start()..span.end()]
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn span_is_two_usizes() {
        assert!(std::mem::size_of::<Span>() <= std::mem::size_of::<usize>() * 2);
        assert_eq!(std::mem::align_of::<Span>(), std::mem::align_of::<usize>());
    }

    #[test]
    fn span_roundtrips() {
        let offsets = [
            0_usize,
            1,
            2,
            3,
            0xff,
            0xffff,
            0xffff_ffff,
            0xffff_ffff_ffff,
        ];
        for doc_id in [0_u32, 1, 2, 3, 0xff, 0xffff, 0xffff_ffff] {
            for start in offsets {
                for end in offsets {
                    if end >= start {
                        let doc = DocId(doc_id);
                        let span = Span::new(doc, start, end);
                        println!("doc={doc_id} start={start} end={end} -> {span:?}");
                        assert_eq!(span.doc(), doc);
                        assert_eq!(span.start(), start);
                        assert_eq!(span.end(), end);
                    }
                }
            }
        }
    }
}

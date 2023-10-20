// RCL -- A sane configuration language.
// Copyright 2023 Ruud van Asseldonk

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// A copy of the License has been included in the root of the repository.

//! The loader is responsible for loading documents.

use std::collections::HashMap;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::{env, path};

use crate::abstraction;
use crate::ast;
use crate::cli::Target;
use crate::cst;
use crate::error::{Error, Result};
use crate::eval::Evaluator;
use crate::lexer;
use crate::parser;
use crate::pprint::{self, concat};
use crate::runtime::{Env, Value};
use crate::source::{Doc, DocId, Span};
use crate::tracer::Tracer;

/// An owned document.
///
/// `Document` is to [`Doc`] what `String` is to `&str`.
pub struct Document {
    /// A friendly name for the source, usually the file path.
    name: String,
    /// The document contents.
    data: String,
}

impl Document {
    pub fn as_doc(&self) -> Doc {
        Doc {
            name: &self.name,
            data: &self.data,
        }
    }
}

#[derive(Debug)]
pub struct PathLookup {
    /// A friendly name, which will be the name of the document.
    name: String,
    /// The path on the file system to load the data from.
    path: PathBuf,
}

/// A filesystem resolves import paths to file contents.
///
/// Importing is split into two stages: first we resolve a path that is
/// referenced from a given document to an absolute path and enforce sandbox
/// policies; then we load from the absolute path.
///
/// NOTE: This design is vulnerable to a TOCTOU issue. Say we canonicalized the
/// path previously and verified that importing it is allowed by the sandbox
/// policy. But now that we are about to open the file, the same path could be
/// a symlink to some file that is *not* allowed by the sandbox policy. Fixing
/// this properly is not possible with the filesystem API in Rust's standard
/// library, it will probably involve using pathfds which are Linux-specific.
/// So fixing this will involve a lot of non-portable unsafe code for an attack
/// that is super specific, and even then, the worst you could do is read a file
/// ... so I am not going to bother handling this properly at this time.
pub trait Filesystem {
    /// Return where to load `path` when imported from file `from`.
    fn resolve(&self, path: &str, from: &str) -> Result<PathLookup>;

    /// Return where to load `path` when that was a CLI argument.
    fn resolve_entrypoint(&self, path: &str) -> Result<PathLookup>;

    /// Load a resolved path from the filesystem.
    fn load(&self, path: PathLookup) -> Result<Document>;
}

/// A dummy filesystem impl to use during initialization.
///
/// This resolves a circular dependency in the error type: to be able to print
/// errors, we need a loader (because errors can reference spans from documents).
/// To have a loader, we need a filesystem. But initializing the filesystem
/// could throw an IO error. There is no actual circular dependency here because
/// the IO error does not reference a document in the loader, but we still need
/// to break the cycle for the type system.
struct PanicFilesystem;

impl Filesystem for PanicFilesystem {
    fn resolve(&self, _: &str, _: &str) -> Result<PathLookup> {
        panic!("Should have initialized the filesystem to a real one before resolving.")
    }
    fn resolve_entrypoint(&self, _: &str) -> Result<PathLookup> {
        panic!("Should have initialized the filesystem to a real one before resolving.")
    }
    fn load(&self, _: PathLookup) -> Result<Document> {
        panic!("Should have initialized the filesystem to a real one before loading.")
    }
}

/// The policy about which documents can be loaded from the filesystem.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub enum SandboxMode {
    #[default]
    Workdir,
    Unrestricted,
}

/// Access the real filesystem, but in a potentially sandboxed manner.
#[derive(Debug)]
pub struct SandboxFilesystem {
    mode: SandboxMode,
    workdir: PathBuf,
}

impl SandboxFilesystem {
    pub fn new(mode: SandboxMode, workdir: Option<&str>) -> io::Result<SandboxFilesystem> {
        let workdir = match workdir {
            Some(d) => PathBuf::from(d),
            None => env::current_dir()?,
        };
        let workdir = fs::canonicalize(workdir)?;
        let result = SandboxFilesystem { mode, workdir };
        Ok(result)
    }

    /// Apply path resolution for an absolute but not yet canonicalized path.
    pub fn resolve_absolute(&self, path_buf: PathBuf) -> Result<PathLookup> {
        // Before we do any sandboxing checks, resolve the file to an absolute
        // path, following symlinks.
        let path_buf = fs::canonicalize(&path_buf).map_err(|err| {
            let fname = path_buf.to_string_lossy().into_owned();
            Error::new(concat! {
                "Failed to access path '"
                pprint::Doc::highlight(&fname).into_owned()
                "': "
                err.to_string()
            })
        })?;

        match self.mode {
            SandboxMode::Unrestricted => {
                // Any path is allowed, nothing to verify.
            }
            SandboxMode::Workdir => {
                if !path_buf.starts_with(&self.workdir) {
                    let fname = path_buf.to_string_lossy().into_owned();
                    let workdir_name = self.workdir.to_string_lossy().into_owned();
                    let mut err = Error::new(concat! {
                        "Sandbox policy '"
                        pprint::Doc::highlight("workdir")
                        "' does not allow loading '"
                        pprint::Doc::highlight(&fname).into_owned()
                        "' because it lies outside of '"
                        pprint::Doc::highlight(&workdir_name).into_owned()
                        "'."
                    });
                    let mut base_dir = self.workdir.clone();
                    while !path_buf.starts_with(&base_dir) {
                        base_dir.pop();
                    }
                    let base_dir_name = base_dir.to_string_lossy().into_owned();
                    err.set_help(concat! {
                        "Try executing from '"
                        pprint::Doc::highlight(&base_dir_name).into_owned()
                        "' or use "
                        pprint::Doc::highlight("--sandbox=unrestricted")
                        "'."
                    });
                    return err.err();
                }
            }
        }

        let friendly_name = if path_buf.starts_with(&self.workdir) {
            let mut result = String::new();
            let mut self_components = path_buf.components();
            // Skip the shared prefix. Note, the zip order is important. If we
            // put self_components firsts, the zip will consume one past the
            // length of `components`. If we put components first, the zip gets
            // a None there, and then we can still call `self_components.next`.
            for _ in self.workdir.components().zip(&mut self_components) {}
            // Then add the path relative to the working directory.
            for component in self_components {
                if !result.is_empty() {
                    result.push('/');
                }
                match component {
                    path::Component::Normal(p) => result.push_str(&p.to_string_lossy()),
                    _ => panic!("Canonicalization and prefix removal should have prevented this."),
                }
            }
            result
        } else {
            // If the path is outside the working directory, we reference it by
            // absolute path.
            path_buf.to_string_lossy().into_owned()
        };

        let result = PathLookup {
            name: friendly_name,
            path: path_buf,
        };
        Ok(result)
    }
}

impl Filesystem for SandboxFilesystem {
    fn resolve(&self, path: &str, from: &str) -> Result<PathLookup> {
        let mut path_buf = self.workdir.clone();

        if path.starts_with("//") {
            // The path is relative to the working directory.
            path_buf.push(Path::new(&path[2..]));
        } else if path.starts_with("/") {
            return Error::new("Importing absolute paths is not allowed.").err();
        } else {
            // The path is relative to the `from` file.
            path_buf.push(from);
            path_buf.pop();
            path_buf.push(path);
        }

        self.resolve_absolute(path_buf)
    }

    fn resolve_entrypoint(&self, path: &str) -> Result<PathLookup> {
        let path_buf: PathBuf = if path.starts_with("/") {
            path.into()
        } else {
            // The path is relative to the working directory.
            let mut path_buf = self.workdir.clone();
            path_buf.push(path);
            path_buf
        };

        self.resolve_absolute(path_buf)
    }

    fn load(&self, path: PathLookup) -> Result<Document> {
        let buf = fs::read_to_string(&path.path).map_err(|err| {
            let fname = path.path.to_string_lossy().into_owned();
            Error::new(concat! {
                "Failed to read from file '"
                pprint::Doc::highlight(&fname).into_owned()
                "': "
                err.to_string()
            })
        })?;

        let doc = Document {
            name: path.name,
            data: buf,
        };

        Ok(doc)
    }
}

pub struct Loader {
    documents: Vec<Document>,

    /// For documents loaded from files, their document id.
    ///
    /// This enables us to avoid loading the same file twice.
    loaded_files: HashMap<PathBuf, DocId>,

    filesystem: Box<dyn Filesystem>,
}

impl Loader {
    pub fn new() -> Loader {
        Loader {
            documents: Vec::new(),
            loaded_files: HashMap::new(),
            filesystem: Box::new(PanicFilesystem),
        }
    }

    /// Enable filesystem access with the given sandbox mode.
    pub fn initialize_filesystem(
        &mut self,
        mode: SandboxMode,
        workdir: Option<&str>,
    ) -> Result<()> {
        let sandbox_fs = SandboxFilesystem::new(mode, workdir).map_err(|err| {
            Error::new(concat! {
                "Failed to initialize filesystem access layer: "
                err.to_string()
            })
        })?;
        self.filesystem = Box::new(sandbox_fs);
        Ok(())
    }

    /// Borrow all documents.
    pub fn as_inputs(&self) -> Vec<Doc> {
        self.documents.iter().map(Document::as_doc).collect()
    }

    /// Borrow a document.
    pub fn get_doc(&self, id: DocId) -> Doc {
        self.documents[id.0 as usize].as_doc()
    }

    /// Return the span that covers the entire document.
    pub fn get_span(&self, id: DocId) -> Span {
        Span::new(id, 0, self.documents[id.0 as usize].data.len())
    }

    /// Lex the given document and return its tokens.
    pub fn get_tokens(&self, id: DocId) -> Result<Vec<lexer::Lexeme>> {
        let doc = self.get_doc(id);
        let tokens = lexer::lex(id, doc.data)?;
        Ok(tokens)
    }

    /// Parse the given document and return its Concrete Syntax Tree.
    pub fn get_cst(&self, id: DocId) -> Result<cst::Prefixed<cst::Expr>> {
        let doc = self.get_doc(id);
        let tokens = self.get_tokens(id)?;
        let (_doc_span, expr) = parser::parse(id, doc.data, &tokens)?;
        Ok(expr)
    }

    /// Parse the given document and return its Abstract Syntax Tree.
    pub fn get_ast(&self, id: DocId) -> Result<ast::Expr> {
        let doc = self.get_doc(id);
        let cst = self.get_cst(id)?;
        let ast = abstraction::abstract_expr(doc.data, &cst)?;
        Ok(ast)
    }

    /// Evaluate the given document and return the resulting value.
    pub fn evaluate(
        &mut self,
        id: DocId,
        env: &mut Env,
        tracer: &mut dyn Tracer,
    ) -> Result<Rc<Value>> {
        let mut evaluator = Evaluator::new(self, tracer);
        evaluator.eval_doc(env, id)
    }

    fn push(&mut self, document: Document) -> DocId {
        let n = self.documents.len();
        self.documents.push(document);
        DocId(n.try_into().expect("Cannot load that many documents!"))
    }

    /// Load stdin into a new document.
    pub fn load_stdin(&mut self) -> Result<DocId> {
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .map_err(|err| Error::new(format!("Failed to read from stdin: {}.", err)))?;
        let doc = Document {
            name: "stdin".to_string(),
            data: buf,
        };
        Ok(self.push(doc))
    }

    /// Load a path that is referenced in the document with name `from`.
    pub fn load_path(&mut self, path: &str, from: Option<DocId>) -> Result<DocId> {
        let from_path = match from {
            Some(id) => self.get_doc(id).name,
            None => "",
        };
        let resolved = self.filesystem.resolve(path, from_path)?;
        assert!(!resolved.name.is_empty());
        self.load_file(resolved)
    }

    /// Load a file into a new document.
    pub fn load_file(&mut self, path: PathLookup) -> Result<DocId> {
        // Avoid loading the same file twice if we already loaded it. This is
        // needed in particular to be able to detect circular imports, because
        // we detect those based on document id.
        if let Some(id) = self.loaded_files.get(&path.path) {
            return Ok(*id);
        }

        let path_buf = path.path.clone();
        let doc = self.filesystem.load(path)?;
        let id = self.push(doc);
        self.loaded_files.insert(path_buf, id);

        Ok(id)
    }

    /// Load a string into a new document.
    pub fn load_string(&mut self, data: String) -> DocId {
        let doc = Document {
            name: "input".to_string(),
            data,
        };
        self.push(doc)
    }

    /// Load the file with the given name, or stdin.
    pub fn load_cli_target(&mut self, target: Target) -> Result<DocId> {
        match target {
            Target::File(fname) => {
                let path = self.filesystem.resolve_entrypoint(&fname)?;
                self.load_file(path)
            }
            Target::Stdin => self.load_stdin(),
        }
    }
}
